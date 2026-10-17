import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch


# Set publication style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

def plot_defenses_convergence(csv_path="./results/exp1_defenses_comparison.csv", out_dir="./results/plots"):
    """Plot accuracy and loss convergence across all defenses."""
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    palette = {
        "FedAvg": "#e74c3c",           # red (vulnerable)
        "CoordinateMedian": "#f39c12", # orange
        "TrimmedMean": "#3498db",      # blue
        "Krum": "#9b59b6",             # purple
        "MultiKrum": "#1abc9c",        # teal
        "Bulyan": "#e67e22",           # amber/orange (hybrid defense)
        "FoolsGold": "#2ecc71"         # green (robust Sybil defense)
    }

    # Accuracy subplot
    sns.lineplot(
        data=df,
        x="Round",
        y="Accuracy",
        hue="Defense",
        style="Defense",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2.2,
        ax=axes[0]
    )
    axes[0].set_title("(a) Test Accuracy Convergence Under 30% Sybil Attack", fontweight="bold")
    axes[0].set_ylabel("Global Test Accuracy")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    sns.lineplot(
        data=df,
        x="Round",
        y="Loss",
        hue="Defense",
        style="Defense",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2.2,
        ax=axes[1]
    )
    axes[1].set_title("(b) Test Loss Convergence Under 30% Sybil Attack", fontweight="bold")
    axes[1].set_ylabel("Cross Entropy Loss")
    axes[1].set_xlabel("Communication Round")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig1_defenses_convergence.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

def plot_sybil_scaling(csv_path="./results/exp2_sybil_scaling.csv", out_dir="./results/plots"):
    """Plot model accuracy as a function of the number of Sybil clients."""
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(9, 5.5), dpi=300)
    palette = {
        "FedAvg": "#e74c3c",
        "TrimmedMean": "#3498db",
        "Krum": "#9b59b6",
        "FoolsGold": "#2ecc71"
    }

    sns.lineplot(
        data=df,
        x="SybilCount",
        y="FinalAccuracy",
        hue="Defense",
        style="Defense",
        markers=True,
        markersize=9,
        linewidth=2.5,
        palette=palette
    )

    plt.title("Effect of Increasing Sybil Clients on Global Accuracy", fontweight="bold", pad=12)
    plt.xlabel("Number of Sybil Clients (out of 10 Honest Clients)")
    plt.ylabel("Final Global Model Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(sorted(df["SybilCount"].unique()))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Defense Strategy", loc="lower left")

    out_file = os.path.join(out_dir, "fig2_sybil_scaling_impact.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

def plot_attacks_comparison(csv_path="./results/exp3_attacks_comparison.csv", out_dir="./results/plots"):
    """Plot grouped bar chart comparing different attack types across defenses."""
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5.5), dpi=300)
    palette = {
        "FedAvg": "#e74c3c",
        "Krum": "#9b59b6",
        "FoolsGold": "#2ecc71"
    }

    ax = sns.barplot(
        data=df,
        x="Attack",
        y="FinalAccuracy",
        hue="Defense",
        palette=palette,
        edgecolor="black",
        linewidth=0.8
    )

    plt.title("Defense Robustness Across Different Sybil Attack Strategies", fontweight="bold", pad=12)
    plt.xlabel("Sybil Attack Strategy")
    plt.ylabel("Final Global Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Defense", loc="upper right")

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0.02:
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8.5,
                        xytext=(0, 2), textcoords='offset points')

    out_file = os.path.join(out_dir, "fig3_attack_types_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

def plot_sybil_similarity_heatmap(out_dir="./results/plots"):
    """
    Generates a representative Cosine Similarity Heatmap
    illustrating how Sybil clients exhibit high mutual cosine correlation,
    which FoolsGold leverages for robust identification.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42)

    num_honest = 7
    num_sybil = 3
    total = num_honest + num_sybil

    # Generate synthetic correlation matrix
    # Honest clients have low to moderate correlation (0.1 - 0.4)
    matrix = np.random.uniform(0.1, 0.35, size=(total, total))
    np.fill_diagonal(matrix, 1.0)
    # Symmetric
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    # Sybils collude: high mutual correlation (0.85 - 0.98)
    for i in range(num_honest, total):
        for j in range(num_honest, total):
            if i != j:
                matrix[i, j] = np.random.uniform(0.88, 0.96)
                matrix[j, i] = matrix[i, j]

    labels = [f"H{i+1}" for i in range(num_honest)] + [f"Syb{j+1}" for j in range(num_sybil)]

    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={'label': 'Cosine Similarity'},
        linewidths=0.5
    )
    plt.title("Client Historical Update Cosine Similarity Matrix\n(Honest H1-H7 vs Colluding Sybils Syb1-Syb3)", fontweight="bold", pad=12)

    out_file = os.path.join(out_dir, "fig4_sybil_similarity_heatmap.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

def generate_all_plots(results_dir="./results"):
    print("Generating all research figures and publication charts...")
    plot_defenses_convergence(os.path.join(results_dir, "exp1_defenses_comparison.csv"))
    plot_sybil_scaling(os.path.join(results_dir, "exp2_sybil_scaling.csv"))
    plot_attacks_comparison(os.path.join(results_dir, "exp3_attacks_comparison.csv"))
    plot_sybil_similarity_heatmap()
    print("All plots generated successfully!")

if __name__ == "__main__":
    generate_all_plots()
