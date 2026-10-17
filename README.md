# Evaluating the Impact of Sybil Clients and Robust Aggregation Defenses on Federated Learning: A Simulation-Based Study

An end-to-end simulation and experimentation framework evaluating **Adaptive Sybil Collusion Attacks** against modern **Robust Aggregation Defenses** in Federated Learning (FL).

---

## 🌟 Overview

Federated Learning (FL) allows distributed edge clients to collaboratively train a centralized global model without sharing raw private data. However, FL is inherently vulnerable to **Sybil attacks**, where an adversary injects multiple colluding, pseudonymous malicious clients into the federated ecosystem.

While conventional Byzantine-robust aggregation algorithms (such as **Krum**, **Trimmed Mean**, and **Median**) rely on Euclidean geometric distances to filter outlier gradient updates, **Adaptive Sybil Attacks** actively evade these defenses by:
1. Constraining the malicious updates to remain within the empirical honest cluster radius (bypassing Euclidean distance tests).
2. Introducing orthogonal diversity jitter across Sybil updates to lower pairwise cosine similarity (bypassing correlation-based defenses like **FoolsGold**).

This repository provides an end-to-end framework featuring:
- **Real-World & Crypto Datasets**: Financial/Crypto Transaction Fraud Detection (284,807 transactions, 30 features, non-IID Dirichlet distribution), Decentralized Anomaly Telemetry, and standard benchmarks.
- **10 Defense Aggregators**: FedAvg, Coordinate-wise Median, Trimmed Mean, Krum, Multi-Krum, Bulyan, FoolsGold, **FLTrust** (NDSS '21 clean-root bootstrapping), **ProofOfTrust** (Stake-weighted slashing defense), and **RFA** (Geometric Median).
- **8 Attack Vectors**: Gaussian Noise, Sign Flipping, Model Scaling, Label Flipping, **Adaptive Sybil Collusion**, **Distributed Backdoor Attack (DBA)**, **Targeted Fraud Evasion**, and **Model Replacement**.
- **High-Performance CUDA & Parallel Engine**: Automatic GPU acceleration and multi-threaded parallel client local training (reducing round latency to <150ms).
- **Dual Security Metrics**: Clean Data Accuracy, Fraud Recall, Precision, F1-Score, and Attack Success Rate (ASR).
- **Interactive Streamlit Web Dashboard**: Real-time simulation, round-by-round convergence tracking, 2D PCA gradient projection, and cosine similarity heatmaps.
- **Comprehensive Benchmark Suite**: Automated multi-experiment runner generating publication-ready figures and CSV metrics.

---

## 📁 Project Architecture

```
sybil/
├── attacks/
│   ├── __init__.py
│   └── sybil_attacks.py       # SybilAttackEngine (noise, sign-flip, scale, label-flip, adaptive, dba, fraud_evasion)
├── data/
│   ├── __init__.py
│   ├── federated_dataset.py   # Real Fraud (284k), Crypto Anomaly, Non-IID Dirichlet partitioners
│   └── raw/creditcard.pt      # Preprocessed 30-feature fraud dataset (34.7 MB)
├── defenses/
│   ├── __init__.py
│   └── aggregators.py         # FedAvg, Median, TrimmedMean, Krum, MultiKrum, Bulyan, FoolsGold, FLTrust, ProofOfTrust, RFA
├── experiments/
│   ├── __init__.py
│   ├── benchmark.py           # Multi-experiment benchmark pipeline (Exp 1, 2, 3)
│   └── plot_results.py        # High-res publication figure generator
├── federated/
│   ├── __init__.py
│   ├── client.py              # FederatedClient (honest & Sybil local training with class weighting)
│   └── server.py              # FederatedServer (multi-threaded parallel training, FLTrust root updates, metrics)
├── models/
│   ├── __init__.py
│   └── networks.py            # Neural network models (FraudClassifier, MLP, CNN) & parameter vector utilities
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
# Note: Provide an explicit target directory name to avoid Windows filesystem trailing-dot errors
git clone https://github.com/devsiraj123cloud/Sybil-Attack-on-Federated-Learning..git Sybil-Attack-on-Federated-Learning
cd Sybil-Attack-on-Federated-Learning
pip install -r requirements.txt
```

### 2. Run Integration Tests

Verify all 5 pipeline stages (models, non-IID data partitioning, aggregators with behavioral assertions, Sybil engine with diversity jitter, FL simulation):

```bash
python main.py --test
# or with pytest:
pytest test_suite.py
```

### 3. Quick CLI Demo

Run a multi-round simulation comparing FedAvg, TrimmedMean, Krum, MultiKrum, Bulyan, and FoolsGold:

```bash
# Run with synthetic data (instant CPU test)
python main.py --demo

# Or run with realistic MNIST benchmark data
python main.py --demo --dataset mnist --rounds 6
```

### 4. Interactive Streamlit Dashboard

Launch the interactive web visualizer:

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

Execute benchmark experiments across defenses, Sybil scaling ratios, and attack vectors:

```bash
# Run full benchmark on synthetic data or MNIST
python main.py --benchmark --dataset synthetic
python main.py --plot
```

---

## 📊 Empirical Findings & Research Analysis

1. **Synthetic vs. Real Image Data Dynamics**:
   - On the synthetic benchmark (linearly separable Gaussian clusters), FedAvg demonstrates surprising resilience to bounded adaptive attacks because the 70% honest majority averages out small perturbations.
   - On realistic non-IID MNIST classification, the non-linear decision boundaries make the global model highly susceptible to poisoned updates, resulting in severe degradation across naive averaging.
2. **Vulnerability of Distance-Based Defenses (Krum & Multi-Krum)**:
   - Because Adaptive Sybils constrain their malicious drift within the empirical radius of the honest cluster, Krum is actively deceived into selecting a poisoned update as the closest neighbor, leading to severe convergence degradation even when FedAvg resists.
3. **Correlation-Based Defense Dynamics (FoolsGold)**:
   - FoolsGold relies on historical cosine similarity tracking. Under standard colluding attacks (identical updates), FoolsGold successfully reduces Sybil influence to near zero.
   - However, when the Adaptive Sybil attacker injects sufficient orthogonal diversity jitter, the pairwise cosine similarity drops below the detection threshold, allowing Sybils to evade suppression.
4. **Dimension-Wise Trimming (Trimmed Mean & Coordinate Median)**:
   - Robust coordinate-wise aggregators remain resilient provided the adversary's budget does not exceed the breakdown point ($f < 50\%$ for Median, $f < \beta$ for Trimmed Mean).

---

## 📄 References & Academic Attribution

- **FoolsGold**: Clement Fung, Chris J. M. Yoon, and Ivan Beschastnikh. *"The Limitations of Federated Learning in Sybil Settings"*, 23rd International Symposium on Research in Attacks, Intrusions and Defenses (**RAID 2020**). [USENIX Link](https://www.usenix.org/conference/raid2020/presentation/fung) | [arXiv:1808.04866](https://arxiv.org/abs/1808.04866).
- **Krum & Multi-Krum**: Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. *"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"*, Advances in Neural Information Processing Systems (**NeurIPS 2017**).
- **Trimmed Mean & Median**: Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. *"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"*, International Conference on Machine Learning (**ICML 2018**).
- **Bulyan**: El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. *"The Hidden Vulnerability of Distributed Learning in Byzantium"*, International Conference on Machine Learning (**ICML 2018**).
- **Recent Work**: Mahdi Ghafourian et al. *"SaFL: Sybil-aware Federated Learning with Application to Face Recognition"*, arXiv:2311.04346, 2023 (revised 2024).
