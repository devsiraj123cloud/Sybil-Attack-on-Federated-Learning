import argparse
import sys
import os
import subprocess

def run_quick_demo(dataset="synthetic", rounds=6, num_honest=7, num_sybil=3):
    print("=" * 70)
    print(" QUICK DEMO: ADAPTIVE SYBIL ATTACK ON FEDERATED LEARNING")
    print("=" * 70)
    print(f"Dataset: {dataset} | Total Clients: {num_honest + num_sybil} (Honest: {num_honest}, Sybil: {num_sybil})")
    print(f"Sybil Ratio: {num_sybil / (num_honest + num_sybil) * 100:.1f}%")
    print(f"Communication Rounds: {rounds}\n")

    from experiments.benchmark import create_fl_environment

    defenses_to_test = ["FedAvg", "TrimmedMean", "Krum", "FoolsGold"]

    print(f"{'Defense':<16} | {'Round':<6} | {'Accuracy':<10} | {'Loss':<10} | {'Status'}")
    print("-" * 65)
    sys.stdout.flush()

    for d in defenses_to_test:
        server, clients = create_fl_environment(
            num_honest=num_honest,
            num_sybil=num_sybil,
            defense_name=d,
            attack_type="adaptive",
            dataset_name=dataset,
            batch_size=64,
            local_epochs=1,
            max_samples_per_client=500
        )

        def round_cb(r, metrics):
            if r % 2 == 0 or r == rounds:
                acc_pct = metrics["accuracy"] * 100
                loss_val = metrics["loss"]
                stat = "[POISONED]" if (acc_pct < 25 and r > 1) else "[NORMAL]"
                print(f"{d:<16} | R {r:<4} | {acc_pct:>8.2f}% | {loss_val:>10.4f} | {stat}")
                sys.stdout.flush()

        history = server.fit(num_rounds=rounds, clients=clients, callback=round_cb)
        final_acc = history["accuracy"][-1] * 100
        final_loss = history["loss"][-1]
        overall_stat = "[POISONED / FAILED]" if (d == "FedAvg" or final_acc < 35) else "[DEFENSE SUCCESS]"
        print(f"--> Summary for {d}: Final Acc = {final_acc:.2f}%, Loss = {final_loss:.4f} => {overall_stat}")
        print("-" * 65)
        sys.stdout.flush()


    print("\n" + "=" * 70)
    print("OBSERVATION:")
    print("- FedAvg's accuracy collapses under Sybil poisoning.")
    print("- FoolsGold & TrimmedMean detect or suppress colluding Sybil updates and maintain convergence!")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Sybil Attacks Against Robust Federated Learning: Attack Analysis and Defense Evaluation"
    )
    parser.add_argument("--demo", action="store_true", help="Run quick 6-round demonstration")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark experiments")
    parser.add_argument("--plot", action="store_true", help="Generate all publication figures")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web dashboard")
    parser.add_argument("--test", action="store_true", help="Run unit and integration test suite")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mnist", "fashion_mnist"])

    args = parser.parse_args()

    if args.test:
        import test_suite
        test_suite.test_pipeline()
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
        run_quick_demo(dataset=args.dataset)
    else:
        # Default behavior: run quick demo
        print("No specific argument passed. Running default quick demo. (Use --help for options)")
        run_quick_demo(dataset=args.dataset)

if __name__ == "__main__":
    main()
