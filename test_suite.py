import sys
import torch
from models.networks import get_model, parameters_to_vector, vector_to_parameters
from data.federated_dataset import get_federated_datasets, get_synthetic_federated_datasets
from defenses.aggregators import (
    FedAvg, CoordinateWiseMedian, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold
)
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient
from federated.server import FederatedServer

def test_pipeline():
    print("=== [1/5] Testing Models & Parameter Conversion ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model("mlp", device=device)
    vec = parameters_to_vector(model)
    print(f"MLP parameter vector length: {vec.shape[0]}")
    vector_to_parameters(vec, model)
    print("Vector to parameters conversion successful!")

    print("\n=== [2/5] Testing Data Partitioning ===")
    client_datasets, test_dataset = get_synthetic_federated_datasets(
        num_clients=5, non_iid=True, num_train=1000, num_test=200
    )
    print(f"Created {len(client_datasets)} client datasets. Test set size: {len(test_dataset)}")

    print("\n=== [3/5] Testing Robust Aggregators ===")
    dummy_updates = [torch.randn(vec.shape[0], device=device) for _ in range(5)]
    # Make one Sybil update an outlier
    dummy_updates[4] = dummy_updates[4] * 100.0

    aggregators = [
        FedAvg(),
        CoordinateWiseMedian(),
        TrimmedMean(beta=0.2),
        Krum(num_byzantine=1),
        MultiKrum(num_byzantine=1),
        Bulyan(num_byzantine=1),
        FoolsGold()
    ]

    for agg in aggregators:
        out, meta = agg.aggregate(dummy_updates, client_ids=list(range(5)))
        assert out.shape == vec.shape, f"{agg.name} output shape mismatch!"
        print(f"Aggregator [{agg.name}] passed. Meta: {list(meta.keys())}")

    print("\n=== [4/5] Testing Sybil Attack Engine ===")
    attack_engine = SybilAttackEngine(attack_type="adaptive")
    honest_updates = dummy_updates[:3]
    sybil_updates = dummy_updates[3:]
    crafted = attack_engine.craft_sybil_updates(
        sybil_client_updates=sybil_updates,
        honest_client_updates=honest_updates
    )
    assert len(crafted) == len(sybil_updates)
    print(f"Adaptive attack generated {len(crafted)} poisoned vectors.")

    print("\n=== [5/5] Testing Full FL Mini Simulation (2 rounds) ===")
    clients = []
    # 3 honest clients
    for i in range(3):
        c = FederatedClient(
            client_id=i,
            dataset=client_datasets[i],
            model_template=model,
            device=device,
            batch_size=32,
            local_epochs=1,
            is_sybil=False
        )
        clients.append(c)

    # 2 Sybil clients
    for i in range(3, 5):
        c = FederatedClient(
            client_id=i,
            dataset=client_datasets[i],
            model_template=model,
            device=device,
            batch_size=32,
            local_epochs=1,
            is_sybil=True
        )
        clients.append(c)

    server = FederatedServer(
        model_template=model,
        test_dataset=test_dataset,
        defense=Krum(num_byzantine=2),
        attack_engine=attack_engine,
        device=device
    )

    history = server.fit(num_rounds=2, clients=clients)
    print(f"Simulation completed! Rounds: {history['round']}, Accuracies: {[round(a, 4) for a in history['accuracy']]}")
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

def test_dashboard_and_configs():
    import os
    assert os.path.exists("app.py"), "app.py dashboard missing!"
    assert os.path.exists("requirements.txt"), "requirements.txt missing!"
    assert os.path.exists("README.md"), "README.md missing!"
    import app
    print("Dashboard and configurations verified successfully!")

if __name__ == "__main__":
    test_pipeline()
    test_dashboard_and_configs()
