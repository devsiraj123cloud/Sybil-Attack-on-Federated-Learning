import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.networks import get_model, parameters_to_vector
from data.federated_dataset import get_federated_datasets, get_synthetic_federated_datasets
from defenses.aggregators import (
    FedAvg, CoordinateWiseMedian, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold, get_aggregator
)
from attacks.sybil_attacks import SybilAttackEngine
from federated.client import FederatedClient
from federated.server import FederatedServer
from experiments.benchmark import create_fl_environment
from experiments.plot_results import (
    plot_defenses_convergence, plot_sybil_scaling, 
    plot_attacks_comparison, plot_sybil_similarity_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Robust FL vs Adaptive Sybil Attacks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-sybil {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-honest {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=50)
st.sidebar.title("Robust FL Benchmark")
st.sidebar.caption("Adaptive Sybil Attacks & Defense Evaluation")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "🚀 Live FL Simulation", 
        "📊 Benchmark Plots & Analysis", 
        "🛡️ Defense Algorithms Deep Dive",
        "🦹 Sybil Attack Inspector"
    ]
)

# ---------------------------------------------------------------------------
# MODE 1: LIVE SIMULATION
# ---------------------------------------------------------------------------
if app_mode == "🚀 Live FL Simulation":
    st.markdown('<div class="main-title">🚀 Interactive Federated Learning Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulate honest and Sybil clients in real time under robust aggregation defenses.</div>', unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.subheader("1. Client Setup")
        num_honest = st.slider("Honest Clients (H)", min_value=2, max_value=15, value=7, step=1)
        num_sybil = st.slider("Sybil / Malicious Clients (S)", min_value=0, max_value=8, value=3, step=1)
        total_clients = num_honest + num_sybil
        sybil_pct = (num_sybil / total_clients) * 100 if total_clients > 0 else 0
        st.info(f"Total: **{total_clients}** clients | Sybil Ratio: **{sybil_pct:.1f}%**")

    with col_cfg2:
        st.subheader("2. Defense Strategy")
        defense_choice = st.selectbox(
            "Aggregator Defense",
            ["FedAvg", "CoordinateMedian", "TrimmedMean", "Krum", "MultiKrum", "Bulyan", "FoolsGold"],
            index=6 # FoolsGold default
        )
        dataset_choice = st.selectbox(
            "Dataset",
            ["synthetic", "mnist", "fashion_mnist"],
            index=0
        )
        rounds = st.slider("Communication Rounds", min_value=1, max_value=15, value=5, step=1)

    with col_cfg3:
        st.subheader("3. Sybil Attack Vector")
        attack_type = st.selectbox(
            "Attack Strategy",
            ["adaptive", "sign_flip", "noise", "model_scale", "label_flip", "honest"],
            index=0
        )
        if attack_type == "adaptive":
            jitter_scale = st.slider("Orthogonal Jitter Scale (Diversity)", 0.0, 1.0, 0.35, 0.05,
                                     help="Perturbation to bypass FoolsGold cosine similarity defense")
            lambda_step = st.slider("Malicious Step (Bounding)", 0.5, 3.0, 1.3, 0.1,
                                    help="Euclidean distance bounding to bypass Krum/TrimmedMean")
        elif attack_type in ["sign_flip", "model_scale", "label_flip"]:
            scale_val = st.slider("Attack Scale Factor", 1.0, 10.0, 3.0, 0.5)
        else:
            jitter_scale, lambda_step, scale_val = 0.35, 1.3, 3.0

    st.markdown("---")

    run_btn = st.button("▶️ Launch Simulation", type="primary", use_container_width=True)

    if run_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # Placeholders for live graphs
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        m_acc = col_m1.empty()
        m_loss = col_m2.empty()
        m_asr = col_m3.empty()
        m_stat = col_m4.empty()

        chart_col1, chart_col2 = st.columns(2)
        chart_acc = chart_col1.empty()
        chart_loss = chart_col2.empty()

        table_placeholder = st.empty()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        status_text.text(f"Initializing FL environment on {device.upper()}...")

        # Setup FL server and clients
        server, clients = create_fl_environment(
            num_honest=num_honest,
            num_sybil=num_sybil,
            defense_name=defense_choice,
            attack_type=attack_type,
            dataset_name=dataset_choice,
            batch_size=64,
            local_epochs=1,
            max_samples_per_client=400,
            device=device
        )

        # Update attack kwargs if adaptive
        if attack_type == "adaptive":
            server.attack_engine.kwargs["jitter_scale"] = jitter_scale
            server.attack_engine.kwargs["lambda_step"] = lambda_step
        elif attack_type in ["sign_flip", "model_scale", "label_flip"]:
            server.attack_engine.kwargs["scale"] = scale_val

        history_records = []
        last_updates = []
        last_client_ids = []

        def round_callback(r, metrics):
            progress = r / rounds
            progress_bar.progress(progress)
            status_text.text(f"Round {r}/{rounds} completed. Evaluating global test accuracy...")

            acc = metrics["accuracy"] * 100
            loss = metrics["loss"]
            asr = metrics["attack_success_rate"] * 100

            m_acc.metric("Global Test Accuracy", f"{acc:.2f}%")
            m_loss.metric("Evaluation Loss", f"{loss:.4f}")
            m_asr.metric("Attack Success Rate", f"{asr:.2f}%")
            
            is_poisoned = acc < 25.0 and r > 1 and num_sybil > 0
            if is_poisoned:
                m_stat.metric("System Health", "POISONED", delta="-Compromised", delta_color="inverse")
            else:
                m_stat.metric("System Health", "CONVERGING", delta="+Normal", delta_color="normal")

            history_records.append({
                "Round": r,
                "Accuracy (%)": acc,
                "Loss": loss,
                "ASR (%)": asr,
                "Status": "POISONED" if is_poisoned else "OK"
            })

            # Update charts
            df_hist = pd.DataFrame(history_records)
            chart_acc.line_chart(df_hist.set_index("Round")[["Accuracy (%)"]])
            chart_loss.line_chart(df_hist.set_index("Round")[["Loss"]])
            table_placeholder.dataframe(df_hist, use_container_width=True)

        history = server.fit(num_rounds=rounds, clients=clients, callback=round_callback)
        progress_bar.progress(1.0)
        status_text.success(f"Simulation Complete! Final Accuracy: {history['accuracy'][-1]*100:.2f}%")

        # Visualizations of client gradients from final round
        st.markdown("### 🔍 Client Update Diagnostics & Sybil Detection")
        vis_col1, vis_col2 = st.columns(2)

        # Retrieve last client updates
        client_updates = []
        client_labels = []
        for c in clients:
            u = c.compute_update(server.global_params)
            client_updates.append(u)
            label = f"Syb-{c.client_id}" if c.is_sybil else f"H-{c.client_id}"
            client_labels.append(label)

        # 1. Cosine similarity heatmap
        with vis_col1:
            st.markdown("#### Cosine Similarity Matrix")
            stacked_u = torch.stack(client_updates, dim=0)
            norm_u = stacked_u / (torch.norm(stacked_u, dim=1, keepdim=True) + 1e-8)
            cos_matrix = torch.mm(norm_u, norm_u.t()).cpu().numpy()

            fig_hm, ax_hm = plt.subplots(figsize=(6, 5), dpi=150)
            sns.heatmap(cos_matrix, xticklabels=client_labels, yticklabels=client_labels,
                        annot=True, fmt=".2f", cmap="vlag", ax=ax_hm, cbar_kws={'label': 'Cosine Sim'})
            ax_hm.set_title("Pairwise Gradient Cosine Similarity")
            st.pyplot(fig_hm)
            plt.close()

        # 2. PCA 2D Gradient Projection
        with vis_col2:
            st.markdown("#### 2D PCA Gradient Projection")
            arr = stacked_u.cpu().numpy()
            if arr.shape[0] >= 2 and arr.shape[1] >= 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(arr)
                fig_pca, ax_pca = plt.subplots(figsize=(6, 5), dpi=150)
                for idx, c in enumerate(clients):
                    color = "#e74c3c" if c.is_sybil else "#2ecc71"
                    marker = "X" if c.is_sybil else "o"
                    ax_pca.scatter(coords[idx, 0], coords[idx, 1], c=color, marker=marker, s=120, edgecolors="black")
                    ax_pca.annotate(client_labels[idx], (coords[idx, 0]+0.1, coords[idx, 1]+0.1), fontsize=9)
                ax_pca.set_title("Client Updates in 2D Feature Space")
                ax_pca.set_xlabel("Principal Component 1")
                ax_pca.set_ylabel("Principal Component 2")
                ax_pca.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_pca)
                plt.close()
            else:
                st.write("Not enough clients for PCA projection.")

# ---------------------------------------------------------------------------
# MODE 2: BENCHMARK PLOTS & ANALYSIS
# ---------------------------------------------------------------------------
elif app_mode == "📊 Benchmark Plots & Analysis":
    st.markdown('<div class="main-title">📊 Benchmark Results & Experimental Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Precomputed research figures and automated benchmark execution.</div>', unsafe_allow_html=True)

    results_dir = "./results"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("⚡ Run Benchmarks Now", type="primary"):
            with st.spinner("Executing full benchmark suite (this takes ~30-45s on CUDA)..."):
                from experiments.benchmark import run_all_benchmarks
                run_all_benchmarks(dataset_name="synthetic", model_name="mlp")
                from experiments.plot_results import generate_all_plots
                generate_all_plots()
            st.success("Benchmarks finished and figures generated!")
            st.rerun()

    tabs = st.tabs([
        "1. Defense Convergence (Exp 1)", 
        "2. Sybil Ratio Scaling (Exp 2)", 
        "3. Attack Strategies (Exp 3)", 
        "4. FoolsGold Correlation Heatmap"
    ])

    fig1_path = os.path.join(plots_dir, "fig1_defenses_convergence.png")
    fig2_path = os.path.join(plots_dir, "fig2_sybil_scaling_impact.png")
    fig3_path = os.path.join(plots_dir, "fig3_attack_types_comparison.png")
    fig4_path = os.path.join(plots_dir, "fig4_sybil_similarity_heatmap.png")

    with tabs[0]:
        st.subheader("Experiment 1: Defense Convergence Under 30% Sybil Attack")
        if os.path.exists(fig1_path):
            st.image(fig1_path, use_column_width=True)
            exp1_csv = os.path.join(results_dir, "exp1_defenses_comparison.csv")
            if os.path.exists(exp1_csv):
                st.dataframe(pd.read_csv(exp1_csv), use_container_width=True)
        else:
            st.warning("Figure not generated yet. Click '⚡ Run Benchmarks Now' to generate it!")

    with tabs[1]:
        st.subheader("Experiment 2: Effect of Increasing Sybil Clients (0 to 5 Sybils)")
        if os.path.exists(fig2_path):
            st.image(fig2_path, use_column_width=True)
            exp2_csv = os.path.join(results_dir, "exp2_sybil_scaling.csv")
            if os.path.exists(exp2_csv):
                st.dataframe(pd.read_csv(exp2_csv), use_container_width=True)
        else:
            st.warning("Figure not generated yet. Click '⚡ Run Benchmarks Now' to generate it!")

    with tabs[2]:
        st.subheader("Experiment 3: Defense Robustness Across Attack Strategies")
        if os.path.exists(fig3_path):
            st.image(fig3_path, use_column_width=True)
            exp3_csv = os.path.join(results_dir, "exp3_attacks_comparison.csv")
            if os.path.exists(exp3_csv):
                st.dataframe(pd.read_csv(exp3_csv), use_container_width=True)
        else:
            st.warning("Figure not generated yet. Click '⚡ Run Benchmarks Now' to generate it!")

    with tabs[3]:
        st.subheader("Experiment 4: FoolsGold Cosine Similarity Detection Mechanism")
        if os.path.exists(fig4_path):
            st.image(fig4_path, use_column_width=True)
        else:
            plot_sybil_similarity_heatmap()
            st.image(fig4_path, use_column_width=True)

# ---------------------------------------------------------------------------
# MODE 3: DEFENSE ALGORITHMS DEEP DIVE
# ---------------------------------------------------------------------------
elif app_mode == "🛡️ Defense Algorithms Deep Dive":
    st.markdown('<div class="main-title">🛡️ Robust Aggregation Defense Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical and algorithmic analysis of Byzantine and Sybil defenses.</div>', unsafe_allow_html=True)

    def_cards = [
        {
            "name": "FedAvg (McMahan et al., 2017)",
            "type": "Standard Baseline",
            "breakdown_point": "0% Sybils (f = 0)",
            "mechanism": "Computes the unweighted or sample-weighted average across all client updates. Highly susceptible to arbitrary parameter poisoning.",
            "vulnerability": "A single unbounded Sybil or a colluding group can skew the global update arbitrarily, halting learning or inducing catastrophic forgetting."
        },
        {
            "name": "Coordinate-wise Median (Yin et al., 2018)",
            "type": "Dimension-wise Median",
            "breakdown_point": "< 50% Byzantine",
            "mechanism": "Computes the median value across clients independently for every coordinate dimension.",
            "vulnerability": "Sub-optimal convergence on non-IID distributions; can suffer dimension-wise drift when Sybils coordinate around marginal medians."
        },
        {
            "name": "Trimmed Mean (Yin et al., 2018)",
            "type": "Coordinate-wise Trimming",
            "breakdown_point": "Beta fraction",
            "mechanism": "Sorts values along each parameter coordinate, strips out the lowest and highest beta fraction, and averages the remainder.",
            "vulnerability": "Requires tuning hyperparameter beta; if the actual Sybil fraction exceeds beta, malicious updates leak into the mean."
        },
        {
            "name": "Krum & Multi-Krum (Blanchard et al., 2017)",
            "type": "Euclidean Distance Minimization",
            "breakdown_point": "f < (n - 2) / 2",
            "mechanism": "Selects the client vector that minimizes the sum of Euclidean distances to its (n - f - 2) nearest neighbors.",
            "vulnerability": "Vulnerable to Adaptive Sybil attacks where Sybils craft updates within the honest cluster diameter, causing Krum to select malicious updates."
        },
        {
            "name": "Bulyan (Mhamdi et al., 2018)",
            "type": "Multi-Krum + Coordinate Trimming",
            "breakdown_point": "f < (n - 3) / 4",
            "mechanism": "First filters theta candidates using Multi-Krum, then applies coordinate-wise trimmed mean on the candidate pool.",
            "vulnerability": "High computational complexity and strict bound on the number of tolerated malicious clients (requires n >= 4f + 3)."
        },
        {
            "name": "FoolsGold (Fung et al., 2020)",
            "type": "Historical Cosine Correlation",
            "breakdown_point": "Robust against arbitrary Sybil counts",
            "mechanism": "Maintains historical gradient memory for each client and computes cosine similarity. Sybils sharing goals exhibit high mutual correlation and are penalized with near-zero aggregation weights.",
            "vulnerability": "Sensitive to feature diversity jitter where Sybils intentionally inject orthogonal noise to lower their mutual cosine similarity."
        }
    ]

    for c in def_cards:
        with st.expander(f"📌 {c['name']} — Breakdown Point: {c['breakdown_point']}", expanded=True):
            st.markdown(f"**Category:** `{c['type']}`")
            st.markdown(f"**Mechanism:** {c['mechanism']}")
            st.markdown(f"**Known Vulnerability / Adaptive Vector:** {c['vulnerability']}")

# ---------------------------------------------------------------------------
# MODE 4: SYBIL ATTACK INSPECTOR
# ---------------------------------------------------------------------------
elif app_mode == "🦹 Sybil Attack Inspector":
    st.markdown('<div class="main-title">🦹 Adaptive Sybil Attack Mechanics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">How modern Sybil attacks circumvent both distance-based and cosine-based defenses.</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### Three Pillars of Adaptive Sybil Collusion:
    
    1. **Opposing Malicious Direction**:
       The attack estimates the honest gradient direction $\mu_{\text{honest}}$ and constructs an adversarial drift vector $v_{\text{dir}} = -\frac{\mu_{\text{honest}}}{\|\mu_{\text{honest}}\|}$.
       
    2. **Euclidean Bounding (Evading Krum & TrimmedMean)**:
       Distance-based aggregators like Krum reject updates that sit far from the main cluster. To bypass this, the Sybils constrain the attack magnitude to stay within the empirical honest cluster radius:
       $$\mathbf{u}_{\text{base}} = \mu_{\text{honest}} + \lambda \cdot R_{\text{honest}} \cdot v_{\text{dir}}$$
       
    3. **Orthogonal Diversity Perturbation (Evading FoolsGold)**:
       FoolsGold penalizes clients with high mutual cosine similarity. To evade detection, each Sybil $i$ injects a randomized perturbation $\xi_i$ projected onto the orthogonal complement of $v_{\text{dir}}$:
       $$\xi_i^{\perp} = \xi_i - (\xi_i \cdot v_{\text{dir}}) v_{\text{dir}}$$
       $$\mathbf{u}_{\text{sybil}, i} = \mathbf{u}_{\text{base}} + \delta \cdot \frac{\xi_i^{\perp}}{\|\xi_i^{\perp}\|}$$
       This lowers the mutual cosine similarity below FoolsGold's penalization threshold while maintaining uniform poisonous drift!
    """)

    st.info("💡 Try adjusting the **Orthogonal Jitter Scale** in the Live Simulation mode to observe the exact threshold where FoolsGold stops penalizing Sybil updates!")
