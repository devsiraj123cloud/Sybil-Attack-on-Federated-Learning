import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
from models.networks import get_model

from data.federated_dataset import get_federated_datasets, get_synthetic_federated_datasets
from defenses.aggregators import (
    FedAvg, CoordinateWiseMedian, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold,
    FLTrust, ProofOfTrust, RFA, get_aggregator
)
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient
from federated.server import FederatedServer

def create_fl_environment(
    num_honest=7,
    num_sybil=3,
    defense_name="fedavg",
    attack_type="adaptive",
    dataset_name="mnist",
    model_name="mlp",
    batch_size=64,
    local_epochs=1,
    lr=0.02,
    max_samples_per_client=600,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    label_flip_target=(7, 1)
):
    """Factory to build server and clients for an experiment run."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_clients = num_honest + num_sybil

    if dataset_name.lower() in ["fraud", "creditcard", "crypto_fraud"]:
        if model_name in ["mlp", "cnn"]:
            model_name = "fraud"
        if label_flip_target == (7, 1):
            label_flip_target = (1, 0)

    # Load datasets with server root support for FLTrust
    client_datasets, test_dataset, server_root_dataset = get_federated_datasets(
        dataset_name=dataset_name,
        num_clients=total_clients,
        non_iid=True,
        dirichlet_alpha=0.5,
        seed=seed,
        max_samples_per_client=max_samples_per_client,
        return_server_root=True
    )

    template_model = get_model(model_name, device=device)

    # Instantiate defense aggregator using flexible factory
    defense = get_aggregator(defense_name, num_byzantine=max(1, num_sybil))

    # Instantiate attack engine
    attack_engine = SybilAttackEngine(
        attack_type=attack_type,
        scale=3.0,
        noise_scale=5.0,
        jitter_scale=0.3,
        lambda_step=1.2,
        total_clients=total_clients
    )

    # Build clients
    clients = []
    for i in range(num_honest):
        client = FederatedClient(
            client_id=i,
            dataset=client_datasets[i],
            model_template=template_model,
            device=device,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=lr,
            is_sybil=False,
            attack_type="none"
        )
        clients.append(client)

    for j in range(num_sybil):
        cid = num_honest + j
        client = FederatedClient(
            client_id=cid,
            dataset=client_datasets[cid],
            model_template=template_model,
            device=device,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=lr,
            is_sybil=True,
            label_flip_target=label_flip_target,
            attack_type=attack_type
        )
        clients.append(client)

    # Build server
    server = FederatedServer(
        model_template=template_model,
        test_dataset=test_dataset,
        server_root_dataset=server_root_dataset,
        defense=defense,
        attack_engine=attack_engine,
        device=device,
        label_flip_target=label_flip_target
    )

    return server, clients

def run_experiment_1_defenses(rounds=15, dataset_name="synthetic", model_name="mlp", output_dir="./results"):
    """
    Experiment 1: Defense comparison under Adaptive Sybil Attack (30% Sybil clients: 7 honest, 3 sybil).
    """
    print("\n" + "="*70)
    print(">> RUNNING EXPERIMENT 1: DEFENSE EVALUATION UNDER ADAPTIVE SYBIL ATTACK")
    print("="*70)

    defenses = ["FedAvg", "CoordinateMedian", "TrimmedMean", "Krum", "MultiKrum", "Bulyan", "FoolsGold", "ProofOfTrust", "FLTrust"]
    results = {}

    all_round_data = []

    for d_name in defenses:
        print(f"\n--> Evaluating Defense: [{d_name}] ...")
        t0 = time.time()
        server, clients = create_fl_environment(
            num_honest=7,
            num_sybil=3,
            defense_name=d_name,
            attack_type="adaptive",
            dataset_name=dataset_name,
            model_name=model_name
        )

        history = server.fit(num_rounds=rounds, clients=clients)
        elapsed = time.time() - t0
        final_acc = history["accuracy"][-1]
        final_loss = history["loss"][-1]
        final_recall = history["fraud_recall"][-1] if len(history["fraud_recall"]) > 0 else 0.0
        print(f"    Completed in {elapsed:.2f}s | Final Acc: {final_acc*100:.2f}% | Fraud Recall: {final_recall*100:.1f}% | Loss: {final_loss:.4f}")

        results[d_name] = {
            "rounds": history["round"],
            "accuracy": history["accuracy"],
            "loss": history["loss"],
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "runtime_sec": elapsed
        }

        for r, acc, loss in zip(history["round"], history["accuracy"], history["loss"]):
            all_round_data.append({
                "Defense": d_name,
                "Round": r,
                "Accuracy": acc,
                "Loss": loss
            })

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_round_data)
    df.to_csv(os.path.join(output_dir, "exp1_defenses_comparison.csv"), index=False)

    return results

def run_experiment_2_sybil_scaling(rounds=10, dataset_name="fraud", model_name="fraud", output_dir="./results"):
    """
    Experiment 2: Effect of increasing Sybil client count (0 to 5 Sybils with 10 honest clients).
    """
    print("\n" + "="*70)
    print(">> RUNNING EXPERIMENT 2: EFFECT OF INCREASING SYBIL CLIENTS")
    print("="*70)

    sybil_counts = [0, 1, 2, 3, 4, 5] # 0% to 33.3%
    defenses = ["FedAvg", "TrimmedMean", "Krum", "FoolsGold", "ProofOfTrust", "FLTrust"]
    scaling_data = []

    for num_sybils in sybil_counts:
        sybil_ratio = num_sybils / (10 + num_sybils)
        print(f"\n--> Testing Sybil Count: {num_sybils} (Ratio: {sybil_ratio*100:.1f}%)")

        for d_name in defenses:
            attack = "adaptive" if num_sybils > 0 else "none"
            server, clients = create_fl_environment(
                num_honest=10,
                num_sybil=num_sybils,
                defense_name=d_name,
                attack_type=attack,
                dataset_name=dataset_name,
                model_name=model_name
            )
            history = server.fit(num_rounds=rounds, clients=clients)
            final_acc = history["accuracy"][-1]
            final_loss = history["loss"][-1]
            print(f"    [{d_name}] -> Final Acc: {final_acc*100:.2f}% | Final Loss: {final_loss:.4f}")

            scaling_data.append({
                "SybilCount": num_sybils,
                "SybilRatio": sybil_ratio,
                "Defense": d_name,
                "FinalAccuracy": final_acc,
                "FinalLoss": final_loss
            })

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(scaling_data)
    df.to_csv(os.path.join(output_dir, "exp2_sybil_scaling.csv"), index=False)
    return df

def run_experiment_3_attacks_comparison(rounds=10, dataset_name="fraud", model_name="fraud", output_dir="./results"):
    """
    Experiment 3: Comparison of different Sybil attack strategies against FedAvg vs FoolsGold vs FLTrust.
    """
    print("\n" + "="*70)
    print(">> RUNNING EXPERIMENT 3: ATTACK STRATEGIES COMPARISON")
    print("="*70)

    attack_types = ["honest", "noise", "sign_flip", "fraud_evasion", "dba", "adaptive"]
    test_defenses = ["FedAvg", "Krum", "FoolsGold", "ProofOfTrust", "FLTrust"]
    attack_data = []

    for atk in attack_types:
        num_s = 0 if atk == "honest" else 3
        print(f"\n--> Testing Attack: [{atk.upper()}] with 3 Sybils ...")

        for d_name in test_defenses:
            server, clients = create_fl_environment(
                num_honest=7,
                num_sybil=num_s,
                defense_name=d_name,
                attack_type=atk,
                dataset_name=dataset_name,
                model_name=model_name
            )
            history = server.fit(num_rounds=rounds, clients=clients)
            final_acc = history["accuracy"][-1]
            final_loss = history["loss"][-1]
            final_asr = history["attack_success_rate"][-1]
            print(f"    [{d_name} vs {atk}] -> Acc: {final_acc*100:.2f}%, ASR: {final_asr*100:.2f}%")

            attack_data.append({
                "Attack": atk,
                "Defense": d_name,
                "FinalAccuracy": final_acc,
                "FinalLoss": final_loss,
                "AttackSuccessRate": final_asr
            })

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(attack_data)
    df.to_csv(os.path.join(output_dir, "exp3_attacks_comparison.csv"), index=False)
    return df

def run_all_benchmarks(dataset_name="synthetic", model_name="mlp"):
    t_start = time.time()
    print("Starting Comprehensive Sybil Attack & Robust FL Benchmark Suite...")
    
    exp1_res = run_experiment_1_defenses(rounds=15, dataset_name=dataset_name, model_name=model_name)
    exp2_res = run_experiment_2_sybil_scaling(rounds=12, dataset_name=dataset_name, model_name=model_name)
    exp3_res = run_experiment_3_attacks_comparison(rounds=12, dataset_name=dataset_name, model_name=model_name)

    total_time = time.time() - t_start
    print("\n" + "="*70)
    print(f"BENCHMARK COMPLETE IN {total_time:.2f} SECONDS!")
    print("Results saved in ./results/")
    print("="*70)

if __name__ == "__main__":
    run_all_benchmarks()
