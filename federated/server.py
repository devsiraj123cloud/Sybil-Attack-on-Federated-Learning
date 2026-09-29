import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.networks import parameters_to_vector, vector_to_parameters
from defenses.aggregators import get_aggregator, BaseAggregator
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient

class FederatedServer:
    """
    Central Federated Learning Server and Orchestrator.
    Manages communication rounds, client aggregation, defense execution, and evaluation.
    """
    def __init__(
        self,
        model_template: nn.Module,
        test_dataset,
        defense: BaseAggregator,
        attack_engine: SybilAttackEngine = None,
        device="cpu",
        server_lr=1.0,
        eval_batch_size=512,
        max_eval_samples=2000,
        label_flip_target=None
    ):
        self.device = device
        self.global_model = copy.deepcopy(model_template).to(device)
        self.test_dataset = test_dataset
        self.defense = defense
        self.attack_engine = attack_engine or SybilAttackEngine(attack_type="none")
        self.server_lr = server_lr
        self.label_flip_target = label_flip_target # tuple (source, target)
        self.max_eval_samples = max_eval_samples

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

        # History tracking
        self.history = {
            "round": [],
            "accuracy": [],
            "loss": [],
            "attack_success_rate": [],
            "defense_name": self.defense.name,
            "metadata": []
        }


    def evaluate(self) -> dict:
        """Evaluates global model on the hold-out test set."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        # For ASR (Attack Success Rate on label flipping)
        asr_target_count = 0
        asr_success_count = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total_samples += x.size(0)

                if self.label_flip_target is not None:
                    src, tgt = self.label_flip_target
                    src_mask = (y == src)
                    if src_mask.sum().item() > 0:
                        asr_target_count += src_mask.sum().item()
                        asr_success_count += (preds[src_mask] == tgt).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)
        asr = (asr_success_count / max(1, asr_target_count)) if asr_target_count > 0 else 0.0

        return {"accuracy": acc, "loss": avg_loss, "attack_success_rate": asr}

    def run_round(self, round_num: int, clients: list) -> dict:
        """
        Executes a single communication round of Federated Learning.
        """
        global_params = parameters_to_vector(self.global_model)

        honest_updates = []
        honest_samples = []
        honest_ids = []

        sybil_raw_updates = []
        sybil_samples = []
        sybil_ids = []

        # 1. Local training on all clients
        for client in clients:
            update, num_samples, _ = client.local_train(global_params)
            if client.is_sybil:
                sybil_raw_updates.append(update)
                sybil_samples.append(num_samples)
                sybil_ids.append(client.client_id)
            else:
                honest_updates.append(update)
                honest_samples.append(num_samples)
                honest_ids.append(client.client_id)

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

        # 3. Server-side robust aggregation defense
        agg_update, meta = self.defense.aggregate(
            updates=all_updates,
            client_weights=all_samples,
            client_ids=all_ids
        )

        # 4. Global parameter update
        new_params = global_params + self.server_lr * agg_update
        vector_to_parameters(new_params, self.global_model)

        # 5. Evaluate post-round metrics
        eval_metrics = self.evaluate()

        # Log history
        self.history["round"].append(round_num)
        self.history["accuracy"].append(eval_metrics["accuracy"])
        self.history["loss"].append(eval_metrics["loss"])
        self.history["attack_success_rate"].append(eval_metrics["attack_success_rate"])
        self.history["metadata"].append(meta)

        return eval_metrics

    def fit(self, num_rounds: int, clients: list, callback=None) -> dict:
        """
        Runs the full FL training loop across `num_rounds`.
        Supports an optional round callback(round_num, metrics) for UI updates.
        """
        # Initial zero-round evaluation
        init_metrics = self.evaluate()
        self.history["round"].append(0)
        self.history["accuracy"].append(init_metrics["accuracy"])
        self.history["loss"].append(init_metrics["loss"])
        self.history["attack_success_rate"].append(init_metrics["attack_success_rate"])
        self.history["metadata"].append({"initial": True})

        if callback is not None:
            callback(0, init_metrics)

        for r in range(1, num_rounds + 1):
            metrics = self.run_round(r, clients)
            if callback is not None:
                callback(r, metrics)

        return self.history
