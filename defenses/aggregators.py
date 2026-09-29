import torch
import numpy as np

class BaseAggregator:
    """Base class for robust federated aggregation algorithms."""
    def __init__(self, name: str):
        self.name = name

    def aggregate(self, updates: list, client_weights: list = None, **kwargs) -> tuple:
        """
        Args:
            updates: list of 1D torch.Tensor representing client parameter updates (w_local - w_global)
            client_weights: optional list of client sample sizes or static weights
        Returns:
            aggregated_update: 1D torch.Tensor
            metadata: dict with defense-specific diagnostics (e.g. chosen indices, scores)
        """
        raise NotImplementedError

class FedAvg(BaseAggregator):
    """
    Standard Federated Averaging (McMahan et al., 2017).
    Vulnerable to Sybil/Byzantine poisoning.
    """
    def __init__(self):
        super(FedAvg, self).__init__("FedAvg")

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n = stacked.shape[0]
        if client_weights is not None:
            w = torch.tensor(client_weights, dtype=torch.float32, device=stacked.device)
            w = w / w.sum()
            aggregated = torch.sum(stacked * w.unsqueeze(1), dim=0)
        else:
            aggregated = torch.mean(stacked, dim=0)
        
        return aggregated, {"method": "FedAvg", "weights": [1.0 / n] * n}

class CoordinateWiseMedian(BaseAggregator):
    """
    Coordinate-wise Median Aggregator (Yin et al., 2018).
    Selects the median value along each parameter coordinate.
    """
    def __init__(self):
        super(CoordinateWiseMedian, self).__init__("CoordinateMedian")

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        median_update, _ = torch.median(stacked, dim=0)
        return median_update, {"method": "CoordinateMedian"}

class TrimmedMean(BaseAggregator):
    """
    Coordinate-wise Trimmed Mean Aggregator (Yin et al., 2018).
    Discards the smallest and largest beta fraction of values per coordinate.
    """
    def __init__(self, beta: float = 0.2):
        super(TrimmedMean, self).__init__(f"TrimmedMean(beta={beta})")
        self.beta = beta

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n = stacked.shape[0]
        k = int(self.beta * n)
        if k == 0:
            return torch.mean(stacked, dim=0), {"method": "TrimmedMean", "trimmed_count": 0}
        
        # Sort along client dimension (dim=0)
        sorted_updates, _ = torch.sort(stacked, dim=0)
        # Slice out lowest k and highest k elements
        trimmed = sorted_updates[k : n - k, :]
        aggregated = torch.mean(trimmed, dim=0)
        return aggregated, {"method": "TrimmedMean", "trimmed_count": k}

class Krum(BaseAggregator):
    """
    Krum Aggregator (Blanchard et al., 2017).
    Selects the single update that minimizes the sum of Euclidean distances 
    to its (n - f - 2) closest neighbors.
    """
    def __init__(self, num_byzantine: int = 2):
        super(Krum, self).__init__(f"Krum(f={num_byzantine})")
        self.num_byzantine = num_byzantine

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n = stacked.shape[0]
        f = min(self.num_byzantine, (n - 3) // 2) if n > 3 else 0
        m = max(1, n - f - 2)

        # Pairwise squared Euclidean distance matrix
        # ||u_i - u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2 <u_i, u_j>
        norms = torch.sum(stacked ** 2, dim=1, keepdim=True)
        dist_matrix = norms + norms.t() - 2.0 * torch.matmul(stacked, stacked.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

        # For each row, sort distances and sum m smallest (excluding self-distance which is 0)
        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        # columns 1 to m (col 0 is distance to self = 0)
        scores = torch.sum(sorted_dists[:, 1 : m + 1], dim=1)

        selected_idx = torch.argmin(scores).item()
        return stacked[selected_idx], {
            "method": "Krum",
            "selected_client": selected_idx,
            "scores": scores.cpu().numpy().tolist()
        }

class MultiKrum(BaseAggregator):
    """
    Multi-Krum Aggregator (Blanchard et al., 2017).
    Iteratively selects top-m candidates with lowest Krum scores and averages them.
    """
    def __init__(self, num_byzantine: int = 2, to_average: int = None):
        super(MultiKrum, self).__init__(f"MultiKrum(f={num_byzantine})")
        self.num_byzantine = num_byzantine
        self.to_average = to_average

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n = stacked.shape[0]
        f = min(self.num_byzantine, (n - 3) // 2) if n > 3 else 0
        m = self.to_average if self.to_average is not None else max(1, n - f)
        m = min(m, n)

        # Compute Krum scores
        neighbor_count = max(1, n - f - 2)
        norms = torch.sum(stacked ** 2, dim=1, keepdim=True)
        dist_matrix = norms + norms.t() - 2.0 * torch.matmul(stacked, stacked.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        scores = torch.sum(sorted_dists[:, 1 : neighbor_count + 1], dim=1)

        # Select top m lowest scores
        _, top_indices = torch.topk(scores, k=m, largest=False)
        selected_updates = stacked[top_indices]
        aggregated = torch.mean(selected_updates, dim=0)

        return aggregated, {
            "method": "MultiKrum",
            "selected_clients": top_indices.cpu().numpy().tolist(),
            "scores": scores.cpu().numpy().tolist()
        }

class Bulyan(BaseAggregator):
    """
    Bulyan Aggregator (Mhamdi et al., 2018).
    Combines Multi-Krum candidate filtering with Coordinate-wise Trimmed Mean.
    """
    def __init__(self, num_byzantine: int = 2):
        super(Bulyan, self).__init__(f"Bulyan(f={num_byzantine})")
        self.num_byzantine = num_byzantine

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0)
        n = stacked.shape[0]
        f = min(self.num_byzantine, (n - 3) // 4) if n > 4 else 0
        theta = n - 2 * f
        if theta <= 0 or theta > n:
            theta = max(1, n - f)

        # Step 1: Multi-Krum selection of theta candidates
        neighbor_count = max(1, n - f - 2)
        norms = torch.sum(stacked ** 2, dim=1, keepdim=True)
        dist_matrix = torch.clamp(norms + norms.t() - 2.0 * torch.matmul(stacked, stacked.t()), min=0.0)
        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        scores = torch.sum(sorted_dists[:, 1 : neighbor_count + 1], dim=1)

        _, candidate_indices = torch.topk(scores, k=theta, largest=False)
        candidates = stacked[candidate_indices] # (theta, d)

        # Step 2: Coordinate-wise trimmed mean on candidate set
        beta = 2 * f
        if beta >= theta:
            beta = max(1, (theta - 1) // 2)

        sorted_cand, _ = torch.sort(candidates, dim=0)
        trimmed = sorted_cand[beta : theta - beta, :]
        if trimmed.shape[0] == 0:
            trimmed = candidates
        aggregated = torch.mean(trimmed, dim=0)

        return aggregated, {
            "method": "Bulyan",
            "candidate_clients": candidate_indices.cpu().numpy().tolist()
        }

class FoolsGold(BaseAggregator):
    """
    FoolsGold Aggregator (Fung, Tian, Yoon, 2018/2020).
    Specifically designed against Sybil Attacks in Federated Learning.
    Tracks cosine similarity between historical client gradient updates.
    Since Sybil clients collude or share identical malicious goals, their updates
    are strongly correlated; FoolsGold penalizes clients with high mutual cosine similarity.
    """
    def __init__(self, memory_decay: float = 0.9, eps: float = 1e-5):
        super(FoolsGold, self).__init__("FoolsGold")
        self.memory_decay = memory_decay
        self.eps = eps
        self.historical_updates = {} # client_id -> 1D tensor

    def aggregate(self, updates: list, client_weights: list = None, client_ids: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n = stacked.shape[0]
        if client_ids is None:
            client_ids = list(range(n))

        # Update historical gradients
        current_history = []
        for i, cid in enumerate(client_ids):
            u = stacked[i]
            if cid in self.historical_updates:
                self.historical_updates[cid] = self.memory_decay * self.historical_updates[cid] + u
            else:
                self.historical_updates[cid] = u.clone()
            current_history.append(self.historical_updates[cid])

        hist_stacked = torch.stack(current_history, dim=0) # (n, d)

        # Compute cosine similarity matrix
        norm_hist = torch.norm(hist_stacked, p=2, dim=1, keepdim=True) + self.eps
        normalized = hist_stacked / norm_hist
        cos_sim = torch.mm(normalized, normalized.t()) # (n, n)
        cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0) # ReLU clipping

        # FoolsGold score formula:
        # For each client i, find maximum similarity to any other client j != i
        sim_copy = cos_sim.clone()
        sim_copy.fill_diagonal_(0.0)
        max_sim, _ = torch.max(sim_copy, dim=1) # (n,)

        # Compute discounting factor alpha_i
        alpha = torch.zeros(n, device=stacked.device)
        for i in range(n):
            for j in range(n):
                if i != j and max_sim[i] > 0:
                    if max_sim[i] > max_sim[j]:
                        sim_copy[i, j] *= (1.0 - max_sim[i]) / (1.0 - max_sim[j] + self.eps)

        v, _ = torch.max(sim_copy, dim=1)
        alpha = 1.0 - v
        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        # Normalize alpha weights
        if alpha.sum() > self.eps:
            alpha = alpha / alpha.sum()
        else:
            alpha = torch.ones(n, device=stacked.device) / n

        aggregated = torch.sum(stacked * alpha.unsqueeze(1), dim=0)

        return aggregated, {
            "method": "FoolsGold",
            "weights": alpha.cpu().numpy().tolist(),
            "cosine_matrix": cos_sim.cpu().numpy().tolist(),
            "max_similarities": max_sim.cpu().numpy().tolist()
        }

def get_aggregator(defense_name: str, num_byzantine: int = 2, **kwargs) -> BaseAggregator:
    """Factory helper to obtain defense aggregator by name."""
    name = defense_name.lower().replace("-", "").replace("_", "")
    if name == "fedavg":
        return FedAvg()
    elif name in ["median", "coordinatewise", "coordinatemedian"]:
        return CoordinateWiseMedian()
    elif name in ["trimmedmean", "trimmed"]:
        beta = kwargs.get("beta", 0.2)
        return TrimmedMean(beta=beta)
    elif name == "krum":
        return Krum(num_byzantine=num_byzantine)
    elif name == "multikrum":
        to_avg = kwargs.get("to_average", None)
        return MultiKrum(num_byzantine=num_byzantine, to_average=to_avg)
    elif name == "bulyan":
        return Bulyan(num_byzantine=num_byzantine)
    elif name == "foolsgold":
        return FoolsGold()
    else:
        raise ValueError(f"Unknown defense aggregator: {defense_name}")
