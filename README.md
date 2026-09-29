# Adaptive Sybil Attacks Against Robust Federated Learning: Attack Analysis & Defense Evaluation

A production-grade research and experimentation framework evaluating **Adaptive Sybil Attacks** against modern **Robust Aggregation Defenses** in Federated Learning (FL).

---

## 🌟 Overview

Federated Learning (FL) allows distributed edge clients to collaboratively train a centralized global model without sharing raw private data. However, FL is inherently vulnerable to **Sybil attacks**, where an adversary injects multiple colluding, pseudonymous malicious clients into the federated ecosystem.

While conventional Byzantine-robust aggregation algorithms (such as **Krum**, **Trimmed Mean**, and **Median**) rely on Euclidean geometric distances to filter outlier gradient updates, **Adaptive Sybil Attacks** actively evade these defenses by:
1. Constraining the malicious updates to remain within the empirical honest cluster radius (bypassing Euclidean distance tests).
2. Introducing orthogonal diversity jitter across Sybil updates to lower pairwise cosine similarity (bypassing correlation-based defenses like **FoolsGold**).

This repository provides an end-to-end framework featuring:
- **7 Defense Aggregators**: FedAvg, Coordinate-wise Median, Trimmed Mean, Krum, Multi-Krum, Bulyan, and FoolsGold.
- **5 Attack Vectors**: Gaussian Noise, Sign Flipping, Model Scaling, Targeted Label Flipping, and Adaptive Sybil Collusion.
- **Interactive Streamlit Web Dashboard**: Real-time simulation, round-by-round convergence tracking, 2D PCA gradient projection, and cosine similarity heatmaps.
- **Comprehensive Benchmark Suite**: Automated multi-experiment runner generating publication-ready figures and CSV metrics.

---

## 📁 Project Architecture

```
sybil/
├── attacks/
│   ├── __init__.py
│   └── sybil_attacks.py       # SybilAttackEngine (noise, sign-flip, scale, label-flip, adaptive)
├── data/
│   ├── __init__.py
│   └── federated_dataset.py   # Non-IID Dirichlet & Synthetic data partitioning
├── defenses/
│   ├── __init__.py
│   └── aggregators.py         # FedAvg, Median, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold
├── experiments/
│   ├── __init__.py
│   ├── benchmark.py           # Multi-experiment benchmark pipeline (Exp 1, 2, 3)
│   └── plot_results.py        # High-res publication figure generator
├── federated/
│   ├── __init__.py
│   ├── client.py              # FederatedClient (honest & Sybil local training)
│   └── server.py              # FederatedServer (orchestrator, defense aggregator, metrics)
├── models/
│   ├── __init__.py
│   └── networks.py            # Neural network models (MLP, CNN) & parameter vector utilities
├── results/
│   ├── plots/                 # Saved PNG figures (convergence, scaling, attack comparison, heatmap)
│   └── *.csv                  # Experiment benchmark data records
├── app.py                     # Interactive Streamlit dashboard
├── main.py                    # Unified CLI entrypoint
├── test_suite.py              # Full unit and integration test suite
└── requirements.txt           # Python dependencies
```

---

## 🛡️ Implemented Defenses

| Aggregator | Category | Tolerance / Breakdown Point | Key Mechanism |
| :--- | :--- | :--- | :--- |
| **FedAvg** | Baseline | 0% (vulnerable) | Standard sample-weighted parameter averaging. |
| **Coordinate Median** | Dimension-wise | $< 50\%$ Byzantine | Coordinate-wise median across client updates. |
| **Trimmed Mean** | Dimension-wise | Parameterized by $\beta$ | Coordinate-wise trimming of top/bottom $\beta$ values. |
| **Krum** | Distance-based | $f < \frac{n - 2}{2}$ | Selects update minimizing distance to $n - f - 2$ closest neighbors. |
| **Multi-Krum** | Distance-based | $f < \frac{n - 2}{2}$ | Averages top-$m$ candidates with lowest Krum scores. |
| **Bulyan** | Hybrid | $f < \frac{n - 3}{4}$ | Multi-Krum candidate filtering followed by coordinate-wise trimming. |
| **FoolsGold** | Correlation-based | Arbitrary Sybil counts | Memory-decay cosine similarity tracking; penalizes correlated Sybils. |

---

## 🦹 Implemented Attack Strategies

1. **Adaptive Sybil Collusion (`adaptive`)**:
   - Direction: Inverted honest gradient direction $\mathbf{v}_{\text{dir}} = -\frac{\boldsymbol{\mu}_{\text{honest}}}{\|\boldsymbol{\mu}_{\text{honest}}\|}$.
   - Distance Bounding: Constrained by honest radius $R_{\text{honest}}$ to remain within Krum's search envelope.
   - Orthogonal Diversity Jitter: $\boldsymbol{\xi}_i^{\perp} = \boldsymbol{\xi}_i - (\boldsymbol{\xi}_i \cdot \mathbf{v}_{\text{dir}})\mathbf{v}_{\text{dir}}$ injected to circumvent FoolsGold's cosine threshold.
2. **Untargeted Gaussian Noise (`noise`)**: Injects high-variance Gaussian perturbations.
3. **Sign Inversion (`sign_flip`)**: Reverses the sign of honest updates ($-\gamma \cdot \boldsymbol{\mu}_{\text{honest}}$).
4. **Model Scaling (`model_scale`)**: Multiplies malicious updates by a large scaling factor to dominate FedAvg.
5. **Data Poisoning / Label Flipping (`label_flip`)**: Locally trains on flipped classification targets (e.g. Class 7 $\to$ Class 1).

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/your-username/sybil.git
cd sybil
pip install -r requirements.txt
```

### 2. Run Integration Tests

Verify all 5 pipeline stages (models, data partitioning, aggregators, Sybil engine, FL simulation):

```bash
python main.py --test
# or with pytest:
pytest test_suite.py
```

### 3. Quick CLI Demo

Run a 6-round simulation comparing FedAvg, TrimmedMean, Krum, and FoolsGold:

```bash
python main.py --demo
```

### 4. Interactive Streamlit Dashboard

Launch the web visualizer:

```bash
python main.py --app
# or directly:
streamlit run app.py
```

Features included:
- **Interactive Live FL Simulation**: Tune client counts, Sybil ratios, defense types, and attack parameters in real time.
- **Gradient Diagnostics**: Pairwise cosine similarity matrix heatmap and 2D PCA update projection.
- **Benchmark Viewer**: Inspect precomputed and live experiment graphs.
- **Algorithmic Deep Dive**: Theoretical reference for all defense and attack formulas.

### 5. Run Full Benchmark Experiments & Generate Figures

Execute all 3 benchmark experiments and render publication figures into `results/plots/`:

```bash
python main.py --benchmark
python main.py --plot
```

---

## 📊 Key Benchmark Findings

1. **FedAvg Vulnerability**: Standard FedAvg degrades rapidly ($< 15\%$ accuracy) under Sybil presence exceeding 20%.
2. **Krum Under Adaptive Sybils**: Because Adaptive Sybils bound their drift within the honest cluster diameter, Krum frequently selects a Sybil update as the "closest neighbor", leading to convergence failure.
3. **FoolsGold Resilience**: By tracking historical update trajectories over communication rounds, FoolsGold reliably isolates and discounts Sybils unless excessive diversity jitter is introduced.
4. **Trimmed Mean Stability**: Trimmed Mean maintains moderate performance provided the Sybil fraction does not exceed the trimming hyperparameter $\beta$.

---

## 📄 Citation & Attribution

If you use this benchmark in academic research or technical evaluations, please reference:
- Fung et al., *"Mitigating Sybils in Federated Learning Poisoning"* (FoolsGold).
- Blanchard et al., *"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"* (Krum).
- Yin et al., *"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"* (Median, Trimmed Mean).
- Mhamdi et al., *"The Hidden Vulnerability of Distributed Learning in Byzantium"* (Bulyan).
