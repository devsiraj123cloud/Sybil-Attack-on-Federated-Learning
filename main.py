import argparse
import sys
import os
import subprocess

def run_quick_demo(dataset="fraud", rounds=6, num_honest=7, num_sybil=3, attack="adaptive"):
    print("=" * 80)
    print(f" QUICK DEMO: {attack.upper()} ATTACK ON FEDERATED LEARNING DEFENSES")
    print("=" * 80)
    print(f"Dataset: {dataset.upper()} | Total Clients: {num_honest + num_sybil} (Honest: {num_honest}, Sybil: {num_sybil})")
    print(f"Sybil Ratio: {num_sybil / (num_honest + num_sybil) * 100:.1f}%")
    print(f"Communication Rounds: {rounds}\n")

    from experiments.benchmark import create_fl_environment

    defenses_to_test = ["FedAvg", "TrimmedMean", "Krum", "MultiKrum", "FoolsGold", "ProofOfTrust", "FLTrust"]
    summary_results = {}

    is_fraud = dataset.lower() in ["fraud", "creditcard", "crypto_fraud"]

    if is_fraud:
        print(f"{'Defense':<14} | {'Round':<6} | {'Accuracy':<10} | {'Fraud Recall':<13} | {'Loss':<9} | {'Status'}")
        print("-" * 80)
    else:
        print(f"{'Defense':<14} | {'Round':<6} | {'Accuracy':<10} | {'Loss':<9} | {'Status'}")
        print("-" * 70)
    sys.stdout.flush()

    for d in defenses_to_test:
        server, clients = create_fl_environment(
            num_honest=num_honest,
            num_sybil=num_sybil,
            defense_name=d,
            attack_type=attack,
            dataset_name=dataset,
            batch_size=64,
            local_epochs=1,
            max_samples_per_client=600
        )

        def round_cb(r, metrics):
            if r % 2 == 0 or r == rounds:
                acc_pct = metrics["accuracy"] * 100
                loss_val = metrics["loss"]
                if is_fraud:
                    recall_pct = metrics.get("fraud_recall", 0.0) * 100
                    stat = "[POISONED]" if (recall_pct < 20.0 and r > 1) else "[SECURE]"
                    print(f"{d:<14} | R {r:<4} | {acc_pct:>8.2f}% | {recall_pct:>11.1f}% | {loss_val:>7.4f} | {stat}")
                else:
                    stat = "[POISONED]" if (acc_pct < 30 and r > 1) else "[NORMAL]"
                    print(f"{d:<14} | R {r:<4} | {acc_pct:>8.2f}% | {loss_val:>7.4f} | {stat}")
                sys.stdout.flush()

        history = server.fit(num_rounds=rounds, clients=clients, callback=round_cb)
        final_acc = history["accuracy"][-1] * 100
        final_loss = history["loss"][-1]
        final_recall = history["fraud_recall"][-1] * 100 if len(history.get("fraud_recall", [])) > 0 else 0.0

        if is_fraud:
            if final_recall < 30.0:
                overall_stat = "[POISONED / EVADED]"
            elif final_recall < 65.0:
                overall_stat = "[DEGRADED DEFENSE]"
            else:
                overall_stat = "[DEFENSE SUCCESS]"
            print(f"--> Summary for {d}: Accuracy = {final_acc:.2f}%, Fraud Recall = {final_recall:.1f}% => {overall_stat}")
        else:
            if final_acc < 35.0:
                overall_stat = "[POISONED / FAILED]"
            elif final_acc < 70.0:
                overall_stat = "[DEGRADED / VULNERABLE]"
            else:
                overall_stat = "[DEFENSE SUCCESS]"
            print(f"--> Summary for {d}: Final Acc = {final_acc:.2f}%, Loss = {final_loss:.4f} => {overall_stat}")

        print("-" * 80)
        sys.stdout.flush()
        summary_results[d] = (final_acc, final_recall if is_fraud else final_acc, overall_stat)

    print("\n" + "=" * 80)
    print("EMPIRICAL OBSERVATION & CRYPTO FEDERATED ANALYSIS:")
    score_idx = 1 if is_fraud else 0
    best_def = max(summary_results.items(), key=lambda x: x[1][score_idx])
    worst_def = min(summary_results.items(), key=lambda x: x[1][score_idx])
    
    metric_label = "Fraud Recall" if is_fraud else "Accuracy"
    print(f"- Highest Performing Defense: {best_def[0]} ({best_def[1][score_idx]:.2f}% {metric_label})")
    print(f"- Lowest Performing Strategy: {worst_def[0]} ({worst_def[1][score_idx]:.2f}% {metric_label})")
    
    for d, (acc, score, stat) in summary_results.items():
        print(f"- {d}: {stat} (Acc: {acc:.1f}%, {metric_label}: {score:.1f}%)")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the Impact of Sybil Clients and Robust Aggregation Defenses on Federated Learning"
    )
    parser.add_argument("--demo", action="store_true", help="Run quick demonstration")
    parser.add_argument("--rounds", type=int, default=6, help="Number of communication rounds (default: 6)")
    parser.add_argument(
        "--attack", 
        type=str, 
        default="adaptive", 
        choices=["adaptive", "noise", "sign_flip", "model_scale", "label_flip", "fraud_evasion", "dba", "model_replacement"], 
        help="Attack type (default: adaptive)"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark experiments")
    parser.add_argument("--plot", action="store_true", help="Generate all publication figures")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web dashboard")
    parser.add_argument("--test", action="store_true", help="Run unit and integration test suite")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="fraud", 
        choices=["fraud", "creditcard", "crypto_fraud", "synthetic", "mnist", "fashion_mnist"],
        help="Dataset to benchmark (default: fraud)"
    )

    args = parser.parse_args()

    if args.test:
        import test_suite
        test_suite.test_pipeline()
        test_suite.test_dashboard_and_configs()
    elif args.benchmark:
        from experiments.benchmark import run_all_benchmarks
        run_all_benchmarks(dataset_name=args.dataset)
        from experiments.plot_results import generate_all_plots
        generate_all_plots()
    elif args.plot:
        from experiments.plot_results import generate_all_plots
        generate_all_plots()
    elif args.app:
        print("Launching Streamlit Web Dashboard...")
        subprocess.run(["streamlit", "run", "app.py"])
    elif args.demo:
        run_quick_demo(dataset=args.dataset, rounds=args.rounds, attack=args.attack)
    else:
        # Default behavior: run quick demo
        print("No specific argument passed. Running default quick demo on real-world Fraud dataset. (Use --help for options)")
        run_quick_demo(dataset=args.dataset, rounds=args.rounds, attack=args.attack)

if __name__ == "__main__":
    main()
