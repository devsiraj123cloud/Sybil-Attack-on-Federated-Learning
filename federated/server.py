import copy
import time
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from models.networks import parameters_to_vector, vector_to_parameters
from defenses.aggregators import get_aggregator, BaseAggregator
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient

class FederatedServer:
    """
    Central Federated Learning Server and Orchestrator.
    Manages communication rounds, client aggregation, defense execution, and evaluation.
    Supports multi-threaded parallel client local training and server root trust bootstrapping.
    """
    def __init__(
        self,
        model_template: nn.Module,
        test_dataset,
        defense: BaseAggregator,
        attack_engine: SybilAttackEngine = None,
        server_root_dataset = None,
        device="cpu",
        server_lr=1.0,
        eval_batch_size=512,
        max_eval_samples=2500,
        label_flip_target=None,
        parallel_clients=True
    ):
        self.device = device
        self.global_model = copy.deepcopy(model_template).to(device)
        self.test_dataset = test_dataset
        self.server_root_dataset = server_root_dataset
        self.defense = defense
        self.attack_engine = attack_engine or SybilAttackEngine(attack_type="none")
        self.server_lr = server_lr
        self.label_flip_target = label_flip_target # tuple (source, target)
        self.max_eval_samples = max_eval_samples
        self.parallel_clients = parallel_clients

        if max_eval_samples is not None and max_eval_samples < len(test_dataset):
            from torch.utils.data import Subset
            self.eval_sub_dataset = Subset(test_dataset, list(range(max_eval_samples)))
        else:
            self.eval_sub_dataset = test_dataset

        self.test_loader = DataLoader(
            self.eval_sub_dataset,
            batch_size=eval_batch_size,
            shuffle=False
        )

        if self.server_root_dataset is not None:
            self.root_loader = DataLoader(
                self.server_root_dataset,
                batch_size=64,
                shuffle=True
            )
        else:
            self.root_loader = None

        # History tracking
        self.history = {
            "round": [],
            "accuracy": [],
            "loss": [],
            "attack_success_rate": [],
            "fraud_recall": [],
            "fraud_precision": [],
            "fraud_f1": [],
            "fraud_asr": [],
            "round_duration": [],
            "defense_name": self.defense.name,
            "metadata": [],
            "last_updates": [],
            "last_client_ids": [],
            "last_client_is_sybil": []
        }

    def compute_server_root_update(self, global_params: torch.Tensor, lr: float = 0.02) -> torch.Tensor:
        """Computes reference clean update g0 on the server hold-out root dataset (for FLTrust)."""
        if self.root_loader is None:
            return None
            
        model_root = copy.deepcopy(self.global_model).to(self.device)
        vector_to_parameters(global_params, model_root)
        model_root.train()
        if getattr(model_root, "fc4", None) is not None and getattr(model_root.fc4, "out_features", None) == 2:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0], device=self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_root.parameters(), lr=lr, momentum=0.9)

        for x, y in self.root_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model_root(x), y)
            loss.backward()
            optimizer.step()

        root_params = parameters_to_vector(model_root)
        return root_params - global_params

    def evaluate(self) -> dict:
        """Evaluates global model on the hold-out test set with detailed fraud and security metrics."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        # Label flip ASR tracking
        asr_target_count = 0
        asr_success_count = 0

        # Tabular Fraud tracking (class 1: fraud, class 0: normal)
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total_samples += x.size(0)

                # Track Label Flip ASR
                if self.label_flip_target is not None:
                    src, tgt = self.label_flip_target
                    src_mask = (y == src)
                    if src_mask.sum().item() > 0:
                        asr_target_count += src_mask.sum().item()
                        asr_success_count += (preds[src_mask] == tgt).sum().item()

                # Binary classification metrics for fraud / anomaly
                y_cpu = y.cpu()
                p_cpu = preds.cpu()
                tp += ((p_cpu == 1) & (y_cpu == 1)).sum().item()
                fp += ((p_cpu == 1) & (y_cpu == 0)).sum().item()
                fn += ((p_cpu == 0) & (y_cpu == 1)).sum().item()
                tn += ((p_cpu == 0) & (y_cpu == 0)).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)
        asr = (asr_success_count / max(1, asr_target_count)) if asr_target_count > 0 else 0.0

        # Fraud detection metrics
        fraud_recall = tp / max(1, tp + fn)
        fraud_precision = tp / max(1, tp + fp)
        fraud_f1 = (2.0 * fraud_precision * fraud_recall) / max(1e-7, fraud_precision + fraud_recall)
        # Fraud Attack Success Rate: fraction of actual frauds misclassified as normal
        fraud_asr = fn / max(1, tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": acc,
            "loss": avg_loss,
            "attack_success_rate": asr,
            "fraud_recall": fraud_recall,
            "fraud_precision": fraud_precision,
            "fraud_f1": fraud_f1,
            "fraud_asr": fraud_asr
        }

    def run_round(self, round_num: int, clients: list) -> dict:
        """Executes a single communication round of Federated Learning."""
        start_t = time.time()
        global_params = parameters_to_vector(self.global_model)

        # 1. Local training on all clients (parallelized for high efficiency)
        if self.parallel_clients and len(clients) > 1 and self.device == "cpu":
            def train_worker(c):
                u, s, m = c.local_train(global_params)
                return (c.client_id, c.is_sybil, u, s, m)

            with ThreadPoolExecutor(max_workers=min(6, len(clients))) as executor:
                results = list(executor.map(train_worker, clients))
        else:
            results = []
            for c in clients:
                u, s, m = c.local_train(global_params)
                results.append((c.client_id, c.is_sybil, u, s, m))

        honest_updates = []
        honest_samples = []
        honest_ids = []
        sybil_raw_updates = []
        sybil_samples = []
        sybil_ids = []

        for cid, is_syb, update, num_samples, _ in results:
            if is_syb:
                sybil_raw_updates.append(update)
                sybil_samples.append(num_samples)
                sybil_ids.append(cid)
            else:
                honest_updates.append(update)
                honest_samples.append(num_samples)
                honest_ids.append(cid)

        # 2. Sybil attack update crafting (attacker collusion)
        if len(sybil_raw_updates) > 0:
            sybil_crafted_updates = self.attack_engine.craft_sybil_updates(
                sybil_client_updates=sybil_raw_updates,
                honest_client_updates=honest_updates,
                global_params=global_params
            )
        else:
            sybil_crafted_updates = []

        # Combine all updates preserving order of clients
        all_updates = []
        all_samples = []
        all_ids = []
        h_idx, s_idx = 0, 0
        for client in clients:
            if client.is_sybil:
                all_updates.append(sybil_crafted_updates[s_idx])
                all_samples.append(sybil_samples[s_idx])
                all_ids.append(client.client_id)
                s_idx += 1
            else:
                all_updates.append(honest_updates[h_idx])
                all_samples.append(honest_samples[h_idx])
                all_ids.append(client.client_id)
                h_idx += 1

        # 3. Server-side reference clean update (for FLTrust)
        server_update = self.compute_server_root_update(global_params)

        # 4. Server-side robust aggregation defense
        agg_update, meta = self.defense.aggregate(
            updates=all_updates,
            client_weights=all_samples,
            client_ids=all_ids,
            server_update=server_update
        )

        # 5. Global parameter update
        new_params = global_params + self.server_lr * agg_update
        vector_to_parameters(new_params, self.global_model)

        self.history["last_updates"] = [
            update.detach().cpu().clone() for update in all_updates
        ]
        self.history["last_client_ids"] = list(all_ids)
        self.history["last_client_is_sybil"] = [
            client.is_sybil for client in clients
        ]
        meta = dict(meta)
        meta["client_update_norms"] = [
            float(torch.norm(update).item()) for update in all_updates
        ]
        meta["client_ids"] = list(all_ids)
        meta["client_is_sybil"] = list(self.history["last_client_is_sybil"])

        round_duration = time.time() - start_t

        # 6. Evaluate post-round metrics
        eval_metrics = self.evaluate()
        eval_metrics["duration"] = round_duration

        # Log history
        self.history["round"].append(round_num)
        self.history["accuracy"].append(eval_metrics["accuracy"])
        self.history["loss"].append(eval_metrics["loss"])
        self.history["attack_success_rate"].append(eval_metrics["attack_success_rate"])
        self.history["fraud_recall"].append(eval_metrics["fraud_recall"])
        self.history["fraud_precision"].append(eval_metrics["fraud_precision"])
        self.history["fraud_f1"].append(eval_metrics["fraud_f1"])
        self.history["fraud_asr"].append(eval_metrics["fraud_asr"])
        self.history["round_duration"].append(round_duration)
        self.history["metadata"].append(meta)

        return eval_metrics

    def fit(self, num_rounds: int, clients: list, callback=None) -> dict:
        """
        Runs the full FL training loop across `num_rounds`.
        Supports an optional round callback(round_num, metrics) for UI updates.
        """
        init_metrics = self.evaluate()
        init_metrics["duration"] = 0.0

        self.history["round"].append(0)
        self.history["accuracy"].append(init_metrics["accuracy"])
        self.history["loss"].append(init_metrics["loss"])
        self.history["attack_success_rate"].append(init_metrics["attack_success_rate"])
        self.history["fraud_recall"].append(init_metrics["fraud_recall"])
        self.history["fraud_precision"].append(init_metrics["fraud_precision"])
        self.history["fraud_f1"].append(init_metrics["fraud_f1"])
        self.history["fraud_asr"].append(init_metrics["fraud_asr"])
        self.history["round_duration"].append(0.0)
        self.history["metadata"].append({"initial": True})

        if callback is not None:
            callback(0, init_metrics)

        for r in range(1, num_rounds + 1):
            metrics = self.run_round(r, clients)
            if callback is not None:
                callback(r, metrics)

        return self.history
