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

DEFENSES = [
    "FLTrust", "ProofOfTrust", "FoolsGold", "MultiKrum", "Krum",
    "Bulyan", "TrimmedMean", "CoordinateMedian", "FedAvg", "RFA"
]
ATTACKS = [
    "adaptive", "fraud_evasion", "dba", "sign_flip", "model_scale",
    "label_flip", "noise", "honest"
]

@st.cache_data(show_spinner=False)
def load_results_csv(path):
    return pd.read_csv(path)

def validate_configuration(defense, total_clients, num_sybil, dataset, attack):
    if defense == "Bulyan" and total_clients < (4 * num_sybil + 3):
        return f"Bulyan requires at least {4 * num_sybil + 3} clients for {num_sybil} Sybils."
    if defense in {"Krum", "MultiKrum"} and total_clients <= num_sybil + 2:
        return "Krum-based defenses need more clients than the Byzantine tolerance permits."
    if defense == "FLTrust" and num_sybil >= total_clients:
        return "FLTrust requires at least one honest client and a trusted root update."
    if dataset == "synthetic" and attack == "fraud_evasion":
        return "fraud_evasion is only meaningful for binary fraud datasets."
    return None

# Page configuration
st.set_page_config(
    page_title="Federated Security Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#f4efff; --muted:#a99fbe; --line:#34264d; --paper:#100b1d; --green:#9b5cff; --lime:#f0a8ff; --blue:#c084fc; --cyan:#64d8ff; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { background:radial-gradient(circle at 12% 0%, #32165a 0, transparent 28rem), radial-gradient(circle at 95% 12%, #172d52 0, transparent 30rem), var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1480px; padding-top:1.35rem; padding-bottom:4rem; }
    .app-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:1rem 1.25rem; margin-bottom:1rem; background:rgba(19,13,35,.82); border:1px solid #4a2f70; border-radius:18px; box-shadow:0 12px 35px rgba(0,0,0,.28), 0 0 30px rgba(155,92,255,.08); }
    .brand { display:flex; align-items:center; gap:.75rem; font-weight:800; letter-spacing:-.04em; font-size:1.25rem; }
    .brand-mark { display:grid; place-items:center; width:2.35rem; height:2.35rem; border-radius:11px; background:linear-gradient(135deg,#8b35ff,#e35cff); color:#fff; font-size:1.25rem; box-shadow:0 0 20px rgba(183,77,255,.5); }
    .header-status { display:flex; align-items:center; gap:.45rem; color:var(--muted); font-size:.78rem; font-family:'DM Mono',monospace; }
    .status-dot { width:.5rem; height:.5rem; border-radius:50%; background:#6dffca; box-shadow:0 0 0 4px rgba(109,255,202,.12), 0 0 12px #6dffca; }
    .section-label { color:var(--blue); font:600 .7rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
    [data-testid="stRadio"] > div { gap:.45rem; }
    [data-testid="stRadio"] label { background:rgba(23,15,39,.78); border:1px solid var(--line); border-radius:10px; padding:.35rem .7rem; }
    .main-title { font-size:2rem; line-height:1.2; letter-spacing:-.055em; font-weight:800; color:var(--ink); margin-bottom:.35rem; }
    .sub-title { font-size:.98rem; color:var(--muted); margin-bottom:1.6rem; }
    h1,h2,h3 { letter-spacing:-.04em; color:var(--ink); }
    h2 { font-size:1.25rem !important; }
    h3 { font-size:1.05rem !important; }
    [data-testid="stMetric"] { background:rgba(24,16,42,.88); border:1px solid #3b2858; border-radius:14px; padding:16px 18px; box-shadow:0 5px 18px rgba(0,0,0,.2), inset 0 1px rgba(255,255,255,.03); }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { font-family:'DM Mono',monospace; color:var(--ink); }
    [data-testid="stTabs"] [role="tab"] { font-weight:600; }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius:12px; }
    .stButton > button[kind="primary"] { background:linear-gradient(135deg,#793dff,#d946ef); border:1px solid #b15cff; border-radius:10px; font-weight:700; color:#fff; box-shadow:0 6px 20px rgba(168,74,255,.3); }
    .stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#965cff,#ec5cff); border-color:#e18aff; }
    [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stAlert"] { border-color:var(--line); }
    [data-baseweb="tab-list"] { background:rgba(19,13,35,.65); border-radius:12px; padding:.25rem; }
    code, .mono { font-family:'DM Mono',monospace; }
    @media (max-width:700px) { .block-container { padding:1.2rem 1rem 3rem; } .main-title { font-size:1.65rem; } }
</style>
""", unsafe_allow_html=True)

# Single-page workspace navigation
st.markdown(
    '<div class="app-header"><div class="brand"><span class="brand-mark">◈</span><span>Federated Security Lab<br><small style="font-size:.68rem;color:#a99fbe;font-weight:500;letter-spacing:.02em">AI-ASSISTED RESEARCH CONSOLE / V1.1</small></span></div><div class="header-status"><span class="status-dot"></span> ENGINE READY&nbsp;&nbsp;|&nbsp;&nbsp; LOCAL MODE</div></div>',
    unsafe_allow_html=True
)

app_mode = st.radio(
    "Workspace section",
    [
        "Live simulation", "Benchmark analysis", "Defense reference", "Attack inspector"
    ],
    format_func=lambda value: {
        "Live simulation": "◉  Live simulation",
        "Benchmark analysis": "▦  Benchmarks",
        "Defense reference": "◈  Defenses",
        "Attack inspector": "⌁  Attack inspector",
    }[value],
    horizontal=True,
    label_visibility="collapsed"
)
st.caption("Select a workspace section above. All experiment controls, results, and diagnostics stay in this single research console.")

# ---------------------------------------------------------------------------
# MODE 1: LIVE SIMULATION
# ---------------------------------------------------------------------------
if app_mode == "Live simulation":
    st.markdown('<div class="main-title">Federated learning, under pressure.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Configure a network, introduce adversarial clients, and observe how aggregation responds.</div>', unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.subheader("01 / Participants")
        num_honest = st.slider("Honest Clients (H)", min_value=2, max_value=15, value=7, step=1)
        num_sybil = st.slider("Sybil / Malicious Clients (S)", min_value=0, max_value=8, value=3, step=1)
        total_clients = num_honest + num_sybil
        sybil_pct = (num_sybil / total_clients) * 100 if total_clients > 0 else 0
        st.info(f"Total: **{total_clients}** clients | Sybil Ratio: **{sybil_pct:.1f}%**")

    with col_cfg2:
        st.subheader("02 / Environment")
        defense_choice = st.selectbox(
            "Aggregator Defense",
            DEFENSES,
            index=0 # FLTrust default
        )
        dataset_choice = st.selectbox(
            "Dataset",
            ["fraud", "crypto_fraud", "synthetic", "mnist", "fashion_mnist"],
            index=0 # Real-world fraud default
        )
        rounds = st.slider("Communication Rounds", min_value=1, max_value=15, value=5, step=1)
        seed = st.number_input("Experiment seed", min_value=0, max_value=999999, value=42, step=1)

    with col_cfg3:
        st.subheader("03 / Adversary")
        attack_type = st.selectbox(
            "Attack Strategy",
            ATTACKS,
            index=0
        )
        if attack_type == "adaptive":
            jitter_scale = st.slider("Orthogonal Jitter Scale (Diversity)", 0.0, 1.0, 0.35, 0.05,
                                     help="Perturbation to bypass FoolsGold cosine similarity defense")
            lambda_step = st.slider("Malicious Step (Bounding)", 0.5, 3.0, 1.3, 0.1,
                                    help="Euclidean distance bounding to bypass Krum/TrimmedMean")
        elif attack_type in ["sign_flip", "model_scale", "label_flip", "dba", "fraud_evasion"]:
            scale_val = st.slider("Attack Scale Factor", 1.0, 10.0, 3.0, 0.5)
        else:
            jitter_scale, lambda_step, scale_val = 0.35, 1.3, 3.0

    st.markdown("---")

    run_btn = st.button("Run simulation", type="primary", use_container_width=True)

    if run_btn:
        configuration_error = validate_configuration(
            defense_choice, total_clients, num_sybil, dataset_choice, attack_type
        )
        if configuration_error:
            st.error(configuration_error)
            st.stop()

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
            device=device,
            seed=int(seed)
        )

        # Update attack kwargs if adaptive
        if attack_type == "adaptive":
            server.attack_engine.kwargs["jitter_scale"] = jitter_scale
            server.attack_engine.kwargs["lambda_step"] = lambda_step
        elif attack_type in ["sign_flip", "model_scale", "label_flip"]:
            server.attack_engine.kwargs["scale"] = scale_val

        history_records = []
        def round_callback(r, metrics):
            progress = r / rounds
            progress_bar.progress(progress)
            status_text.text(f"Round {r}/{rounds} completed. Evaluating global test accuracy...")

            acc = metrics["accuracy"] * 100
            loss = metrics["loss"]
            asr = metrics["attack_success_rate"] * 100
            recall = metrics.get("fraud_recall", 0.0) * 100
            is_fraud_ds = dataset_choice in ["fraud", "crypto_fraud"]

            m_acc.metric("Global Accuracy", f"{acc:.2f}%")
            m_loss.metric("Evaluation Loss", f"{loss:.4f}")
            if is_fraud_ds:
                m_asr.metric("Fraud Recall", f"{recall:.1f}%",
                             delta=f"F1 {metrics.get('fraud_f1', 0.0) * 100:.1f}%")
            else:
                m_asr.metric("Attack Success Rate", f"{asr:.2f}%")
            
            is_poisoned = (
                metrics.get("fraud_asr", 0.0) > 0.8
                if is_fraud_ds else metrics.get("attack_success_rate", 0.0) > 0.8
            ) and r > 0 and num_sybil > 0
            if is_poisoned:
                m_stat.metric("Defense Status", "VULNERABLE / POISONED", delta="-Attack Succeeded", delta_color="inverse")
            else:
                m_stat.metric("Defense Status", "ROBUST / SECURE", delta="+Consensus Protected", delta_color="normal")

            rec = {
                "Round": r,
                "Accuracy (%)": acc,
                "Loss": loss,
                "Status": "POISONED" if is_poisoned else "OK"
            }
            if is_fraud_ds:
                rec["Fraud Recall (%)"] = recall
                rec["Fraud ASR (%)"] = metrics.get("fraud_asr", 0.0) * 100
            else:
                rec["ASR (%)"] = asr
            history_records.append(rec)

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

        # Use the exact crafted updates that were aggregated in the final round.
        client_updates = history.get("last_updates", [])
        client_ids = history.get("last_client_ids", [])
        client_is_sybil = history.get("last_client_is_sybil", [])
        client_labels = [
            f"Syb-{cid}" if is_sybil else f"H-{cid}"
            for cid, is_sybil in zip(client_ids, client_is_sybil)
        ]
        if not client_updates:
            st.info("No client updates were recorded for diagnostics.")
            st.stop()

        # 1. Cosine similarity heatmap
        with vis_col1:
            st.markdown("#### Cosine Similarity Matrix")
            stacked_u = torch.stack(client_updates, dim=0)
            norm_u = stacked_u / (torch.norm(stacked_u, dim=1, keepdim=True) + 1e-8)
            cos_matrix = torch.mm(norm_u, norm_u.t()).cpu().numpy()

            fig_hm, ax_hm = plt.subplots(figsize=(6, 5), dpi=150)
            sns.heatmap(cos_matrix, xticklabels=client_labels, yticklabels=client_labels,
                        annot=True, fmt=".2f", cmap="magma", ax=ax_hm, cbar_kws={'label': 'Cosine Sim'})
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
                for idx, is_sybil in enumerate(client_is_sybil):
                    color = "#ff4f9a" if is_sybil else "#64d8ff"
                    marker = "X" if is_sybil else "o"
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
elif app_mode == "Benchmark analysis":
    st.markdown('<div class="main-title">Benchmark analysis</div>', unsafe_allow_html=True)
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
                st.dataframe(load_results_csv(exp1_csv), use_container_width=True)
        else:
            st.warning("Figure not generated yet. Click '⚡ Run Benchmarks Now' to generate it!")

    with tabs[1]:
        st.subheader("Experiment 2: Effect of Increasing Sybil Clients (0 to 5 Sybils)")
        if os.path.exists(fig2_path):
            st.image(fig2_path, use_column_width=True)
            exp2_csv = os.path.join(results_dir, "exp2_sybil_scaling.csv")
            if os.path.exists(exp2_csv):
                st.dataframe(load_results_csv(exp2_csv), use_container_width=True)
        else:
            st.warning("Figure not generated yet. Click '⚡ Run Benchmarks Now' to generate it!")

    with tabs[2]:
        st.subheader("Experiment 3: Defense Robustness Across Attack Strategies")
        if os.path.exists(fig3_path):
            st.image(fig3_path, use_column_width=True)
            exp3_csv = os.path.join(results_dir, "exp3_attacks_comparison.csv")
            if os.path.exists(exp3_csv):
                st.dataframe(load_results_csv(exp3_csv), use_container_width=True)
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
elif app_mode == "Defense reference":
    st.markdown('<div class="main-title">Defense reference</div>', unsafe_allow_html=True)
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
        ,
        {
            "name": "FLTrust (Cao et al., 2021)",
            "type": "Trusted Root Update",
            "breakdown_point": "Depends on root-data quality",
            "mechanism": "Scores each client update by its positive cosine similarity with a clean server-root update, then normalizes trusted magnitudes.",
            "vulnerability": "A weak, unrepresentative, or contaminated root dataset can produce a misleading trust direction."
        },
        {
            "name": "ProofOfTrust",
            "type": "Stake-weighted Trust",
            "breakdown_point": "Depends on stake distribution",
            "mechanism": "Combines update trust with client stake and reduces the influence of clients whose updates repeatedly disagree with the trusted direction.",
            "vulnerability": "Large stake concentration can amplify a compromised identity unless stake is distributed and auditable."
        },
        {
            "name": "RFA (Pillutla et al., 2019)",
            "type": "Geometric Median",
            "breakdown_point": "Robust geometric center",
            "mechanism": "Uses an iterative robust geometric median to reduce the effect of extreme client updates without requiring a trusted root.",
            "vulnerability": "Repeated high-dimensional distance calculations can be expensive and non-IID honest updates may be treated as outliers."
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
elif app_mode == "Attack inspector":
    st.markdown('<div class="main-title">Attack inspector</div>', unsafe_allow_html=True)
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
