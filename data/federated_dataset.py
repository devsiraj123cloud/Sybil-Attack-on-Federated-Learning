import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

class TensorDatasetWrapper(Dataset):
    """Simple wrapper for in-memory tensors."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_federated_datasets(
    dataset_name="mnist",
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    data_dir="./data/raw",
    seed=42,
    **kwargs
):

    """
    Partitions dataset across `num_clients`.
    Supports 'mnist', 'fashion_mnist', and 'synthetic'.
    Returns:
        client_datasets: list of Dataset objects (one per client)
        test_dataset: Dataset object for global server evaluation
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"[Warning] Failed to download MNIST: {e}. Falling back to Synthetic dataset.")
            return get_synthetic_federated_datasets(num_clients, non_iid, dirichlet_alpha, seed)

    elif dataset_name.lower() == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        try:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"[Warning] Failed to download FashionMNIST: {e}. Falling back to Synthetic dataset.")
            return get_synthetic_federated_datasets(num_clients, non_iid, dirichlet_alpha, seed)

    else:
        return get_synthetic_federated_datasets(num_clients, non_iid, dirichlet_alpha, seed)

    # Extract targets for partitioning
    if hasattr(train_dataset, 'targets'):
        targets = np.array(train_dataset.targets)
    else:
        targets = np.array([y for _, y in train_dataset])

    num_samples = len(train_dataset)
    num_classes = 10

    if not non_iid:
        # IID partitioning: randomly shuffle indices and assign equally
        indices = np.random.permutation(num_samples)
        client_splits = np.array_split(indices, num_clients)
    else:
        # Non-IID Dirichlet distribution over classes
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            # Dirichlet proportions
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            # Rescale proportions based on remaining items
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for c in range(num_clients):
                client_indices[c].extend(splits[c])

        client_splits = [np.array(idx, dtype=int) for idx in client_indices]

    max_samples_per_client = kwargs.get("max_samples_per_client", 1000)
    if max_samples_per_client is not None and max_samples_per_client > 0:
        client_splits = [
            split[:max_samples_per_client] if len(split) > max_samples_per_client else split
            for split in client_splits
        ]

    client_datasets = [Subset(train_dataset, split) for split in client_splits]
    return client_datasets, test_dataset


def get_synthetic_federated_datasets(
    num_clients=10,
    non_iid=True,
    dirichlet_alpha=0.5,
    seed=42,
    num_train=6000,
    num_test=1000,
    num_features=784,
    num_classes=10
):
    """
    Generates synthetic benchmark data mirroring image feature distributions.
    Useful for instant deterministic testing and environments with restricted internet access.
    """
    rng = np.random.RandomState(seed)

    # Class prototypes
    prototypes = rng.randn(num_classes, num_features).astype(np.float32)

    def generate_split(n_samples):
        y = rng.randint(0, num_classes, size=n_samples)
        noise = rng.randn(n_samples, num_features).astype(np.float32) * 0.8
        X = prototypes[y] + noise
        # Reshape to (N, 1, 28, 28)
        X = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

    X_train, y_train = generate_split(num_train)
    X_test, y_test = generate_split(num_test)

    train_dataset = TensorDatasetWrapper(X_train, y_train)
    test_dataset = TensorDatasetWrapper(X_test, y_test)

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
    return client_datasets, test_dataset
