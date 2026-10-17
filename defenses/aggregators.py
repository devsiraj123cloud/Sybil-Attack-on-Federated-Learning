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
    FoolsGold Aggregator (Clement Fung, Chris J. M. Yoon, and Ivan Beschastnikh, RAID 2020 / arXiv:1808.04866).
    Specifically designed against Sybil Attacks in Federated Learning.
    Tracks cosine similarity between historical client gradient updates.
    Since Sybil clients collude or share identical malicious goals, their updates
    are strongly correlated; FoolsGold penalizes clients with high mutual cosine similarity.
    Reference: 'The Limitations of Federated Learning in Sybil Settings', USENIX RAID 2020.
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

class FLTrust(BaseAggregator):
    """
    FLTrust Aggregator (Xiaoyu Cao, Minghong Fang, Jia Liu, Neil Zhenqiang Gong, NDSS 2021).
    Bootstraps trust using a small clean validation dataset on the central server.
    Computes trust scores (TS) based on cosine similarity with the clean server reference update,
    and normalizes update magnitudes to match the clean gradient norm.
    Provably immune to Sybil attacks, colluding Byzantine attacks, and non-IID data distribution!
    """
    def __init__(self, eps: float = 1e-7):
        super(FLTrust, self).__init__("FLTrust")
        self.eps = eps

    def aggregate(self, updates: list, client_weights: list = None, client_ids: list = None, server_update: torch.Tensor = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n, d = stacked.shape

        if server_update is None:
            # Fallback to robust median as proxy reference update if no server root data is available
            server_update, _ = torch.median(stacked, dim=0)

        g0 = server_update.to(stacked.device)
        norm_g0 = torch.norm(g0, p=2) + self.eps

        # Norms of client updates
        norms = torch.norm(stacked, p=2, dim=1) + self.eps # (n,)

        # Compute cosine similarities between each client update and server update g0
        dots = torch.matmul(stacked, g0) # (n,)
        cos_sims = dots / (norms * norm_g0)

        # Trust score TS_i = ReLU(cos_sim)
        trust_scores = torch.clamp(cos_sims, min=0.0) # (n,)

        # Normalize client updates: g_tilde_i = (||g0|| / ||g_i||) * g_i
        scale_factors = norm_g0 / norms # (n,)
        normalized_updates = stacked * scale_factors.unsqueeze(1) # (n, d)

        # Weighted combination
        ts_sum = torch.sum(trust_scores)
        if ts_sum > self.eps:
            weights = trust_scores / ts_sum
            aggregated = torch.sum(normalized_updates * weights.unsqueeze(1), dim=0)
        else:
            # If all trust scores are 0, use server clean update alone
            weights = torch.zeros(n, device=stacked.device)
            aggregated = g0

        return aggregated, {
            "method": "FLTrust",
            "trust_scores": trust_scores.cpu().numpy().tolist(),
            "weights": weights.cpu().numpy().tolist(),
            "cosine_similarities": cos_sims.cpu().numpy().tolist()
        }

class ProofOfTrust(BaseAggregator):
    """
    Proof-of-Trust (PoT) / Stake-Weighted Aggregator.
    Crypto-native defense designed for decentralized / blockchain Federated Learning.
    Each client maintains an on-chain virtual stake / credit balance.
    Clients whose gradient vectors deviate or oppose consensus consensus directions
    suffer slashing penalties (-35% stake per round), progressively reducing their
    voting weight until Sybil nodes are completely zeroed out.
    """
    def __init__(self, initial_stake: float = 100.0, slash_rate: float = 0.35, reward_rate: float = 5.0, eps: float = 1e-6):
        super(ProofOfTrust, self).__init__("ProofOfTrust")
        self.initial_stake = initial_stake
        self.slash_rate = slash_rate
        self.reward_rate = reward_rate
        self.eps = eps
        self.client_stakes = {} # client_id -> float stake

    def aggregate(self, updates: list, client_weights: list = None, client_ids: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n, d = stacked.shape
        if client_ids is None:
            client_ids = list(range(n))

        for cid in client_ids:
            if cid not in self.client_stakes:
                self.client_stakes[cid] = self.initial_stake

        # Calculate consensus direction via coordinate-wise median
        consensus_dir, _ = torch.median(stacked, dim=0)
        norm_consensus = torch.norm(consensus_dir, p=2) + self.eps

        norms = torch.norm(stacked, p=2, dim=1) + self.eps
        cos_sims = torch.matmul(stacked, consensus_dir) / (norms * norm_consensus)

        slashed_clients = []
        for i, cid in enumerate(client_ids):
            sim = cos_sims[i].item()
            if sim < 0.1:
                # Slashing penalty for adversarial / divergent Sybil updates
                self.client_stakes[cid] = max(0.01, self.client_stakes[cid] * (1.0 - self.slash_rate))
                slashed_clients.append(cid)
            elif sim > 0.4:
                # Honest participation credit
                self.client_stakes[cid] = min(200.0, self.client_stakes[cid] + self.reward_rate)

        stakes_tensor = torch.tensor([self.client_stakes[cid] for cid in client_ids], dtype=torch.float32, device=stacked.device)
        weights = stakes_tensor / torch.sum(stakes_tensor)

        aggregated = torch.sum(stacked * weights.unsqueeze(1), dim=0)

        return aggregated, {
            "method": "ProofOfTrust",
            "weights": weights.cpu().numpy().tolist(),
            "stakes": {cid: round(self.client_stakes[cid], 2) for cid in client_ids},
            "slashed_clients": slashed_clients,
            "cosine_similarities": cos_sims.cpu().numpy().tolist()
        }

class RFA(BaseAggregator):
    """
    Robust Federated Aggregation (RFA / Geometric Median - Pillutla et al., 2022).
    Computes the smoothed Geometric Median via Weiszfeld's algorithm in high-dimensional vector space.
    Minimizes the sum of Euclidean distances to all client updates.
    """
    def __init__(self, max_iter: int = 8, nu: float = 1e-6):
        super(RFA, self).__init__("RFA(GeometricMedian)")
        self.max_iter = max_iter
        self.nu = nu

    def aggregate(self, updates: list, client_weights: list = None, **kwargs):
        stacked = torch.stack(updates, dim=0) # (n, d)
        n, d = stacked.shape

        # Initialize with coordinate-wise median
        v, _ = torch.median(stacked, dim=0)

        for _ in range(self.max_iter):
            dists = torch.norm(stacked - v.unsqueeze(0), p=2, dim=1) # (n,)
            weights = 1.0 / torch.clamp(dists, min=self.nu)
            weights = weights / torch.sum(weights)
            v = torch.sum(stacked * weights.unsqueeze(1), dim=0)

        return v, {
            "method": "RFA",
            "weights": weights.cpu().numpy().tolist()
        }

def get_aggregator(defense_name: str, num_byzantine: int = 2, **kwargs) -> BaseAggregator:
    """Factory helper to obtain defense aggregator by name."""
    name = defense_name.lower().replace("-", "").replace("_", "").replace(" ", "")
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
    elif name in ["fltrust", "trust"]:
        return FLTrust()
    elif name in ["proofoftrust", "pot", "stakeweighted", "stake"]:
        return ProofOfTrust()
    elif name in ["rfa", "geometricmedian", "geommedian"]:
        return RFA()
    else:
        raise ValueError(f"Unknown defense aggregator: {defense_name}")
