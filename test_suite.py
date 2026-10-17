import sys
import torch
from models.networks import get_model, parameters_to_vector, vector_to_parameters, FraudClassifier
from data.federated_dataset import get_federated_datasets, get_synthetic_federated_datasets
from defenses.aggregators import (
    FedAvg, CoordinateWiseMedian, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold, FLTrust, ProofOfTrust, RFA, get_aggregator
)
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient
from federated.server import FederatedServer

def test_pipeline():
    print("=== [1/6] Testing Models & Parameter Conversion ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test FraudClassifier
    fraud_model = get_model("fraud", device=device, input_dim=30, num_classes=2)
    dummy_input = torch.randn(4, 30, device=device)
    dummy_out = fraud_model(dummy_input)
    assert dummy_out.shape == (4, 2), f"FraudClassifier output shape mismatch: {dummy_out.shape}"
    vec_fraud = parameters_to_vector(fraud_model)
    print(f"FraudClassifier verified! Parameter vector length: {vec_fraud.shape[0]}")

    model = get_model("mlp", device=device)
    vec = parameters_to_vector(model)
    print(f"MLP parameter vector length: {vec.shape[0]}")
    vector_to_parameters(vec, model)
    print("Vector to parameters conversion successful!")

    print("\n=== [2/6] Testing Real-World Fraud Data Partitioning ===")
    fraud_clients, fraud_test, fraud_root = get_federated_datasets(
        dataset_name="fraud",
        num_clients=3,
        non_iid=True,
        max_samples_per_client=500,
        return_server_root=True
    )
    print(f"Created {len(fraud_clients)} fraud client datasets. Test set size: {len(fraud_test)}. Root set size: {len(fraud_root)}")
    assert len(fraud_clients) == 3
    assert len(fraud_test) > 0
    assert len(fraud_root) > 0

    print("\n=== [3/6] Testing Robust Aggregators (FLTrust & ProofOfTrust) ===")
    dummy_updates = [torch.randn(vec.shape[0], device=device) for _ in range(5)]
    dummy_updates[4] = dummy_updates[4] * 100.0 # Extreme Sybil outlier

    aggregators = [
        FedAvg(),
        CoordinateWiseMedian(),
        TrimmedMean(beta=0.2),
        Krum(num_byzantine=1),
        MultiKrum(num_byzantine=1),
        Bulyan(num_byzantine=1),
        FoolsGold(),
        FLTrust(),
        ProofOfTrust(),
        RFA()
    ]

    for agg in aggregators:
        server_ref = torch.randn(vec.shape[0], device=device)
        agg_vec, meta = agg.aggregate(dummy_updates, client_ids=[0, 1, 2, 3, 4], server_update=server_ref)
        assert agg_vec.shape == vec.shape, f"Aggregator {agg.name} returned shape mismatch: {agg_vec.shape}"
        assert not torch.isnan(agg_vec).any(), f"Aggregator {agg.name} returned NaNs!"

    # Behavioral assertion for FoolsGold Sybil penalization:
    fg = FoolsGold()
    h1 = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0]*(vec.shape[0]-4), device=device)
    h2 = torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0]*(vec.shape[0]-4), device=device)
    s1 = torch.tensor([0.0, 0.0, 1.0, 1.0] + [0.0]*(vec.shape[0]-4), device=device)
    s2 = torch.tensor([0.0, 0.0, 1.0, 1.0] + [0.0]*(vec.shape[0]-4), device=device)
    _, fg_meta = fg.aggregate([h1, h2, s1, s2], client_ids=[0, 1, 2, 3])
    assert fg_meta["weights"][2] < fg_meta["weights"][0], "FoolsGold failed to penalize colluding Sybils!"
    print("FoolsGold Sybil suppression assertion passed (Sybil weight < Honest weight)!")

    # Behavioral assertion for FLTrust:
    flt = FLTrust()
    ref_g0 = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0]*(vec.shape[0]-4), device=device)
    adv_s = torch.tensor([-1.0, 0.0, 0.0, 0.0] + [0.0]*(vec.shape[0]-4), device=device) # opposing direction
    _, flt_meta = flt.aggregate([h1, adv_s], server_update=ref_g0)
    assert flt_meta["trust_scores"][1] == 0.0, "FLTrust failed to zero out opposing malicious update!"
    assert flt_meta["trust_scores"][0] > 0.0, "FLTrust failed to grant trust score to aligned honest update!"
    print("FLTrust assertion passed (Malicious Trust Score = 0, Honest Trust Score > 0)!")

    print("\n=== [4/6] Testing Advanced Sybil Attack Vectors ===")
    # 1. Sign-flip attack assertion
    sign_flip_engine = SybilAttackEngine(attack_type="sign_flip", scale=2.0)
    honest_updates = dummy_updates[:3]
    sybil_updates = dummy_updates[3:]
    sf_crafted = sign_flip_engine.craft_sybil_updates(sybil_updates, honest_updates)
    mean_h = torch.mean(torch.stack(honest_updates), dim=0)
    dot_prod = torch.dot(sf_crafted[0], mean_h).item()
    assert dot_prod < 0, "Sign flip attack failed to reverse honest update direction!"
    print(f"Sign-flip attack verified: dot product with honest mean is negative ({dot_prod:.2f})")

    # 2. Adaptive attack assertion
    attack_engine = SybilAttackEngine(attack_type="adaptive", jitter_scale=0.4, lambda_step=1.2)
    crafted = attack_engine.craft_sybil_updates(
        sybil_client_updates=sybil_updates,
        honest_client_updates=honest_updates
    )
    cos_sim = torch.cosine_similarity(crafted[0].unsqueeze(0), crafted[1].unsqueeze(0)).item()
    assert cos_sim < 0.99, f"Adaptive jitter failed to diversify Sybil updates (cos_sim={cos_sim})!"
    print(f"Adaptive attack verified: Sybil mutual cos_sim={cos_sim:.4f} (< 0.99)")

    # 3. Distributed Backdoor Attack (DBA)
    dba_engine = SybilAttackEngine(attack_type="dba", scale=2.5)
    dba_crafted = dba_engine.craft_sybil_updates(sybil_updates, honest_updates)
    assert len(dba_crafted) == len(sybil_updates), "DBA crafted count mismatch"
    print("Distributed Backdoor Attack (DBA) engine verified!")

    print("\n=== [5/6] Testing Full Federated Learning Mini Simulation on Fraud Data ===")
    fraud_clients_objs = []
    for i in range(2):
        c = FederatedClient(
            client_id=i,
            dataset=fraud_clients[i],
            model_template=fraud_model,
            device=device,
            batch_size=32,
            local_epochs=1,
            is_sybil=False
        )
        fraud_clients_objs.append(c)

    # 1 Sybil client with fraud evasion attack
    sybil_c = FederatedClient(
        client_id=2,
        dataset=fraud_clients[2],
        model_template=fraud_model,
        device=device,
        batch_size=32,
        local_epochs=1,
        is_sybil=True,
        attack_type="fraud_evasion"
    )
    fraud_clients_objs.append(sybil_c)

    server = FederatedServer(
        model_template=fraud_model,
        test_dataset=fraud_test,
        server_root_dataset=fraud_root,
        defense=FLTrust(),
        attack_engine=attack_engine,
        device=device,
        max_eval_samples=500
    )

    history = server.fit(num_rounds=2, clients=fraud_clients_objs)
    print(f"Simulation completed! Rounds: {history['round']}")
    print(f"Accuracy: {[round(a, 4) for a in history['accuracy']]}")
    print(f"Fraud Recall: {[round(r, 4) for r in history['fraud_recall']]}")
    print(f"Round Durations: {[round(d, 3) for d in history['round_duration']]}s")

    print("\n=== [6/6] All Pipelines, Defenses, and Datasets Verified ===")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

def test_dashboard_and_configs():
    import os
    assert os.path.exists("app.py"), "app.py dashboard missing!"
    assert os.path.exists("requirements.txt"), "requirements.txt missing!"
    assert os.path.exists("README.md"), "README.md missing!"
    with open("app.py", "r", encoding="utf-8") as f:
        compile(f.read(), "app.py", "exec")
    print("Dashboard and configurations verified successfully!")

if __name__ == "__main__":
    test_pipeline()
    test_dashboard_and_configs()
