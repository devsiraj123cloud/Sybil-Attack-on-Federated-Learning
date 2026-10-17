import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

class TensorDatasetWrapper(Dataset):
    """
    High-performance in-memory Tensor Dataset wrapper.
    Supports both indexed access and fast direct tensor slicing.
    """
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long)
            
        self.data = data
        self.targets = targets
        self.tensors = (self.data, self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_fraud_federated_datasets(
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    data_dir="./data/raw",
    seed=42,
    num_train_normals=30000,
    num_test_normals=5000,
    max_samples_per_client=1500,
    return_server_root=False
):
    """
    Loads and partitions the real-world CreditCard / Financial Transaction Fraud Dataset.
    Features: 30 numerical PCA and scaled attributes (V1-V28, Time, Amount).
    Labels: 0 (Normal Transaction), 1 (Fraudulent Transaction).
    
    Partitions non-IID across banks/validator nodes using Dirichlet distributions on fraud density.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pt_path = os.path.join(data_dir, "creditcard.pt")
    
    if os.path.exists(pt_path):
        saved = torch.load(pt_path)
        all_X = saved["features"]
        all_y = saved["targets"]
    else:
        print("[Info] Downloading CreditCardFraudDetection dataset from OpenML...")
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler
        data = fetch_openml("CreditCardFraudDetection", version=1, as_frame=True, parser="auto")
        df = data.frame
        features = [c for c in df.columns if c != "Class"]
        X_np = df[features].values.astype(np.float32)
        y_np = df["Class"].astype(int).values
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np)
        
        all_X = torch.tensor(X_np, dtype=torch.float32)
        all_y = torch.tensor(y_np, dtype=torch.long)
        os.makedirs(data_dir, exist_ok=True)
        torch.save({
            "features": all_X,
            "targets": all_y,
            "num_features": 30,
            "num_classes": 2
        }, pt_path)
        print(f"[Info] Saved {pt_path}")

    fraud_indices = (all_y == 1).nonzero(as_tuple=True)[0].numpy()
    normal_indices = (all_y == 0).nonzero(as_tuple=True)[0].numpy()

    np.random.shuffle(fraud_indices)
    np.random.shuffle(normal_indices)

    # 1. Test set allocation (100 fraud + num_test_normals normal)
    n_test_fraud = min(100, len(fraud_indices) // 4)
    test_fraud_idx = fraud_indices[:n_test_fraud]
    test_norm_idx = normal_indices[:num_test_normals]
    test_idx = np.concatenate([test_fraud_idx, test_norm_idx])
    np.random.shuffle(test_idx)

    test_X = all_X[test_idx]
    test_y = all_y[test_idx]
    test_dataset = TensorDatasetWrapper(test_X, test_y)

    # 2. Server root dataset for FLTrust (clean hold-out with 40 fraud + 200 normal)
    rem_fraud = fraud_indices[n_test_fraud:]
    rem_normal = normal_indices[num_test_normals:]

    n_root_fraud = min(40, len(rem_fraud) // 6)
    root_fraud_idx = rem_fraud[:n_root_fraud]
    root_norm_idx = rem_normal[:200]
    root_idx = np.concatenate([root_fraud_idx, root_norm_idx])
    np.random.shuffle(root_idx)
    server_root_dataset = TensorDatasetWrapper(all_X[root_idx], all_y[root_idx])

    # 3. Client training distribution
    train_fraud = rem_fraud[n_root_fraud:]
    train_normal = rem_normal[200 : 200 + num_train_normals]

    client_indices = [[] for _ in range(num_clients)]

    if not non_iid:
        # IID distribution
        f_splits = np.array_split(train_fraud, num_clients)
        n_splits = np.array_split(train_normal, num_clients)
        for c in range(num_clients):
            client_indices[c] = np.concatenate([f_splits[c], n_splits[c]])
    else:
        # Non-IID Dirichlet partitioning on fraud (some clients observe fraud spikes)
        # and on normal transactions
        proportions_f = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        f_split_pts = (np.cumsum(proportions_f) * len(train_fraud)).astype(int)[:-1]
        f_splits = np.split(train_fraud, f_split_pts)

        proportions_n = np.random.dirichlet(np.repeat(dirichlet_alpha * 2, num_clients))
        n_split_pts = (np.cumsum(proportions_n) * len(train_normal)).astype(int)[:-1]
        n_splits = np.split(train_normal, n_split_pts)

        for c in range(num_clients):
            client_c_idx = np.concatenate([f_splits[c], n_splits[c]])
            np.random.shuffle(client_c_idx)
            client_indices[c] = client_c_idx

    # Apply max_samples_per_client cap
    client_datasets = []
    for c in range(num_clients):
        c_idx = client_indices[c]
        if max_samples_per_client is not None and len(c_idx) > max_samples_per_client:
            # Preserve fraud examples when trimming
            c_y = all_y[c_idx].numpy()
            c_fraud = c_idx[c_y == 1]
            c_norm = c_idx[c_y == 0]
            max_norm = max(1, max_samples_per_client - len(c_fraud))
            c_idx = np.concatenate([c_fraud, c_norm[:max_norm]])
            np.random.shuffle(c_idx)

        c_X = all_X[c_idx]
        c_y = all_y[c_idx]
        client_datasets.append(TensorDatasetWrapper(c_X, c_y))

    if return_server_root:
        return client_datasets, test_dataset, server_root_dataset
    return client_datasets, test_dataset


def get_crypto_fraud_datasets(
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    seed=42,
    num_samples=12000,
    num_features=30,
    return_server_root=False
):
    """
    High-fidelity Synthetic Crypto / Blockchain Transaction Fraud Dataset generator.
    Simulates decentralized network features:
      - Gas price, gas used, transaction amount, contract call complexity
      - Graph topological centrality (in-degree, out-degree, clustering coeff)
      - Wash trading cycle score, mixer interaction indicator, flash loan flag
    """
    rng = np.random.RandomState(seed)
    
    # 5% synthetic fraud / sybil wash transactions
    labels = (rng.rand(num_samples) < 0.05).astype(np.int64)
    
    # Base normal transaction distributions
    features = rng.randn(num_samples, num_features).astype(np.float32)
    
    # Correlate fraud transactions with specific anomaly signatures
    fraud_mask = (labels == 1)
    features[fraud_mask, 0] += 3.2  # Gas price spike
    features[fraud_mask, 1] *= 2.5  # Transaction velocity
    features[fraud_mask, 5] += 4.0  # Mixer interaction flag
    features[fraud_mask, 12] -= 2.8 # Centrality deviation
    features[fraud_mask, 28] += 3.5 # Wash trading indicator

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # 20% hold-out test set
    n_test = int(0.2 * num_samples)
    indices = rng.permutation(num_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    test_dataset = TensorDatasetWrapper(X[test_idx], y[test_idx])

    # Server root dataset (100 samples for FLTrust)
    root_idx = train_idx[:100]
    train_client_idx = train_idx[100:]
    server_root_dataset = TensorDatasetWrapper(X[root_idx], y[root_idx])

    train_labels = y[train_client_idx].numpy()
    client_indices = [[] for _ in range(num_clients)]

    if not non_iid:
        splits = np.array_split(train_client_idx, num_clients)
        client_splits = [train_client_idx[s] for s in splits]
    else:
        for k in [0, 1]:
            idx_k = train_client_idx[train_labels == k]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for c in range(num_clients):
                client_indices[c].extend(splits[c])
        client_splits = [np.array(idx, dtype=int) for idx in client_indices]

    client_datasets = [TensorDatasetWrapper(X[s], y[s]) for s in client_splits]

    if return_server_root:
        return client_datasets, test_dataset, server_root_dataset
    return client_datasets, test_dataset


def get_synthetic_federated_datasets(
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    seed=42,
    num_train=6000,
    num_test=1000,
    num_features=784,
    num_classes=10,
    return_server_root=False
):
    """
    Generates synthetic prototype benchmark data mirroring image feature distributions.
    """
    rng = np.random.RandomState(seed)
    prototypes = rng.randn(num_classes, num_features).astype(np.float32)

    def generate_split(n_samples):
        y = rng.randint(0, num_classes, size=n_samples)
        noise = rng.randn(n_samples, num_features).astype(np.float32) * 0.8
        X = prototypes[y] + noise
        X = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

    X_train, y_train = generate_split(num_train)
    X_test, y_test = generate_split(num_test)

    train_dataset = TensorDatasetWrapper(X_train, y_train)
    test_dataset = TensorDatasetWrapper(X_test, y_test)

    # Server root dataset (100 samples)
    root_dataset = TensorDatasetWrapper(X_train[:100], y_train[:100])

    targets = y_train.numpy()
    if not non_iid:
        indices = rng.permutation(num_train)
        client_splits = np.array_split(indices, num_clients)
    else:
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for c in range(num_clients):
                client_indices[c].extend(splits[c])
        client_splits = [np.array(idx, dtype=int) for idx in client_indices]

    client_datasets = [Subset(train_dataset, split) for split in client_splits]
    if return_server_root:
        return client_datasets, test_dataset, root_dataset
    return client_datasets, test_dataset


def get_federated_datasets(
    dataset_name="fraud",
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    data_dir="./data/raw",
    seed=42,
    return_server_root=False,
    **kwargs
):
    """
    Unified factory for federated datasets.
    Supported:
      - 'fraud' / 'creditcard': Real-world 30-feature Financial/Crypto Fraud Dataset.
      - 'crypto_fraud': Synthetic Decentralized Transaction Anomaly Dataset.
      - 'mnist': MNIST 28x28 digits.
      - 'fashion_mnist': Fashion-MNIST 28x28 clothing.
      - 'synthetic': 10-class prototype synthetic generator.
    """
    d_name = dataset_name.lower()
    
    if d_name in ["fraud", "creditcard"]:
        return get_fraud_federated_datasets(
            num_clients=num_clients,
            non_iid=non_iid,
            dirichlet_alpha=dirichlet_alpha,
            data_dir=data_dir,
            seed=seed,
            return_server_root=return_server_root,
            **kwargs
        )
    elif d_name in ["crypto_fraud", "crypto"]:
        return get_crypto_fraud_datasets(
            num_clients=num_clients,
            non_iid=non_iid,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
            return_server_root=return_server_root,
            **kwargs
        )
    elif d_name == "synthetic":
        return get_synthetic_federated_datasets(
            num_clients=num_clients,
            non_iid=non_iid,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
            return_server_root=return_server_root,
            **kwargs
        )
    elif d_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"[Warning] Failed to load MNIST: {e}. Falling back to synthetic.")
            return get_synthetic_federated_datasets(num_clients, non_iid, dirichlet_alpha, seed, return_server_root=return_server_root)
    elif d_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        try:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"[Warning] Failed to load FashionMNIST: {e}. Falling back to synthetic.")
            return get_synthetic_federated_datasets(num_clients, non_iid, dirichlet_alpha, seed, return_server_root=return_server_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standard image partition logic for MNIST / FashionMNIST
    if hasattr(train_dataset, "targets"):
        targets = np.array(train_dataset.targets)
    else:
        targets = np.array([y for _, y in train_dataset])

    num_samples = len(train_dataset)
    num_classes = 10

    if not non_iid:
        indices = np.random.permutation(num_samples)
        client_splits = np.array_split(indices, num_clients)
    else:
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for c in range(num_clients):
                client_indices[c].extend(splits[c])
        client_splits = [np.array(idx, dtype=int) for idx in client_indices]

    max_samples = kwargs.get("max_samples_per_client", 1000)
    if max_samples is not None and max_samples > 0:
        client_splits = [s[:max_samples] if len(s) > max_samples else s for s in client_splits]

    client_datasets = [Subset(train_dataset, split) for split in client_splits]
    server_root_dataset = Subset(train_dataset, list(range(200)))

    if return_server_root:
        return client_datasets, test_dataset, server_root_dataset
    return client_datasets, test_dataset
