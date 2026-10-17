import torch
import numpy as np

class SybilAttackEngine:
    """
    Sybil and Byzantine attack engine for Robust Federated Learning.
    
    Implements:
      1. 'noise': Untargeted Gaussian/random gradient poisoning.
      2. 'sign_flip': Sign inversion of honest gradients (-gamma * mu).
      3. 'label_flip': Data poisoning (source class -> target class).
      4. 'model_scale': Malicious update scaling to overpower aggregators.
      5. 'adaptive': Adaptive collusion attack with distance bounding (evading Krum/Median)
                    and orthogonal diversity jitter (evading FoolsGold cosine detection).
    """
    def __init__(self, attack_type="adaptive", **kwargs):
        self.attack_type = attack_type.lower()
        self.kwargs = kwargs

    def craft_sybil_updates(
        self,
        sybil_client_updates: list,
        honest_client_updates: list = None,
        global_params: torch.Tensor = None,
        target_direction: torch.Tensor = None
    ) -> list:
        num_sybils = len(sybil_client_updates)
        if num_sybils == 0:
            return []

        device = sybil_client_updates[0].device
        dim = sybil_client_updates[0].shape[0]

        if self.attack_type in ["none", "honest"]:
            return sybil_client_updates

        # Estimate honest distribution statistics if available
        if honest_client_updates is not None and len(honest_client_updates) > 0:
            stacked_honest = torch.stack(honest_client_updates, dim=0) # (H, d)
            mean_honest = torch.mean(stacked_honest, dim=0)
            std_honest = torch.std(stacked_honest, dim=0) + 1e-6
            honest_dists = torch.norm(stacked_honest - mean_honest, dim=1)
            max_honest_dist = torch.max(honest_dists).item()
            mean_honest_norm = torch.norm(mean_honest).item()
        else:
            mean_honest = torch.zeros(dim, device=device)
            std_honest = torch.ones(dim, device=device)
            max_honest_dist = 1.0
            mean_honest_norm = 1.0

        if self.attack_type == "noise":
            noise_scale = self.kwargs.get("noise_scale", 5.0) * max(mean_honest_norm, 0.5)
            poisoned = []
            for _ in range(num_sybils):
                noise = torch.randn(dim, device=device) * noise_scale
                poisoned.append(noise)
            return poisoned

        elif self.attack_type == "sign_flip":
            scale = self.kwargs.get("scale", 3.0)
            poisoned = []
            for _ in range(num_sybils):
                # Reverse the honest direction
                poisoned.append(-scale * mean_honest)
            return poisoned

        elif self.attack_type == "model_scale":
            scale = self.kwargs.get("scale", 6.0)
            poisoned = []
            for u in sybil_client_updates:
                poisoned.append(scale * u)
            return poisoned

        elif self.attack_type == "label_flip":
            # Sybils trained on flipped labels locally; updates are preserved or scaled
            scale = self.kwargs.get("scale", 2.0)
            return [scale * u for u in sybil_client_updates]

        elif self.attack_type == "adaptive":
            """
            Adaptive Sybil Attack:
            1. Direction: Malicious drift opposing honest convergence or steering towards loss increase.
            2. Distance constraint: Bounded within honest Euclidean radius (to deceive Krum/TrimmedMean).
            3. Diversity perturbation: Sybils inject controlled orthogonal noise so mutual cosine similarity
               is lowered below FoolsGold's high-similarity suppression threshold!
            """
            # Target poisoned drift: opposite of honest direction
            v_dir = -mean_honest / (torch.norm(mean_honest) + 1e-8)

            # Malicious magnitude: bounded by the honest cluster diameter to stay within Krum's search radius
            lambda_step = self.kwargs.get("lambda_step", 1.5)
            effective_radius = max(max_honest_dist, mean_honest_norm * 1.5)
            base_malicious = mean_honest + lambda_step * effective_radius * v_dir

            jitter_scale = self.kwargs.get("jitter_scale", 0.4)
            poisoned = []
            for i in range(num_sybils):
                # Generate random perturbation vector
                xi = torch.randn(dim, device=device)
                # Project out component along v_dir to maintain net malicious direction
                proj = torch.dot(xi, v_dir) * v_dir
                ortho_jitter = xi - proj
                ortho_norm = torch.norm(ortho_jitter) + 1e-8
                ortho_jitter = (ortho_jitter / ortho_norm) * (jitter_scale * effective_radius)

                # Individualized Sybil update
                sybil_vec = base_malicious + ortho_jitter
                poisoned.append(sybil_vec)

            return poisoned

        elif self.attack_type in ["distributed_backdoor", "dba"]:
            """
            Distributed Backdoor Attack (DBA - Xie et al., ICLR 2020):
            Decomposes global backdoor triggers across multiple Sybil agents.
            Each Sybil modifies only a disjoint slice of model parameters / features,
            preventing pairwise cosine detection (defeats FoolsGold) and staying within
            normal distance bounds (defeats Krum).
            """
            poisoned = []
            slice_size = dim // max(1, num_sybils)
            target_scale = self.kwargs.get("scale", 2.5)

            for i in range(num_sybils):
                # Start with honest mean update
                sybil_vec = mean_honest.clone()
                start_idx = i * slice_size
                end_idx = min(dim, (i + 1) * slice_size)

                # Inject localized targeted backdoor perturbation in this Sybil's partition
                sybil_vec[start_idx:end_idx] = -target_scale * sybil_vec[start_idx:end_idx]
                poisoned.append(sybil_vec)

            return poisoned

        elif self.attack_type in ["fraud_evasion", "targeted_evasion"]:
            """
            Targeted Fraud Evasion Attack:
            Sybils craft updates to suppress fraud detection sensitivity (label 1 -> 0),
            allowing fraudulent transactions to bypass federated screening.
            Bridges honest direction with fraud suppression vector.
            """
            target_dir = -mean_honest if target_direction is None else target_direction
            scale = self.kwargs.get("scale", 3.0)
            poisoned = []
            for u in sybil_client_updates:
                evasion_vec = u + scale * target_dir
                poisoned.append(evasion_vec)
            return poisoned

        elif self.attack_type == "model_replacement":
            """
            Constrain-and-Scale Model Replacement (Bagdasaryan et al., AISTATS 2020):
            Amplifies malicious updates by gamma = (total_clients / num_sybils)
            to dominate and overwrite the global model in a single aggregation round.
            """
            total_clients = self.kwargs.get("total_clients", num_sybils * 3)
            gamma = float(total_clients) / float(max(1, num_sybils))
            return [gamma * u for u in sybil_client_updates]

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
