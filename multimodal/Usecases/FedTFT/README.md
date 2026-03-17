# FedTFT — Federated Temporal Fusion Transformer for Multi-Horizon Psychiatric Risk Prediction

---

## Overview

FedTFT is a federated learning framework for predicting imminent dangerous actions (1-hour, 1-day, 1-week horizons) from wearable sensor data collected across multiple hospitals, without sharing patient data.

Two core contributions — **neither exists in the federated psychiatric risk prediction literature prior to this work**:

---

### 1. Horizon-Decoupled Federated Prediction (HDFP) — *architectural novelty*

**The problem it solves:** Standard multi-horizon models use a single shared output layer (64→3) that simultaneously optimises loss for all three prediction horizons. Because imminent risk at 1 hour is driven by acute physiological signals (e.g., sudden heart rate spikes), risk at 1 day by circadian disruption, and risk at 1 week by longer contextual and behavioural patterns, the gradient signals from each horizon conflict with one another inside the shared layer. This cross-horizon gradient interference degrades the shared backbone's representation, forcing it to compromise between three temporally incompatible objectives.

**What HDFP does:** Replaces the single shared output layer (64→3) with a shared gated residual feature transform (64→64) followed by three independent linear output projections (64→1 each), one per prediction horizon. *(Note: the manuscript abstract uses the shorthand "gated residual head per horizon" to refer collectively to this shared transform + per-horizon linear projection pathway; the precise architecture is as described here and in `model_fedtft_hdfp.py`.)* Each head receives gradients exclusively from its own horizon loss — so the 1h head learns from acute physiological dynamics, the 1d head from circadian patterns, and the 1w head from contextual trends, without any interference. All parameters (backbone + all three heads) remain fully federated across hospitals every round — no personalisation, no parameter exclusion.

**Why it is novel:** No prior federated learning study in psychiatric risk prediction has applied horizon-decoupled output heads to multi-horizon prediction from continuous wearable sensor streams.

---

### 2. Federated Volume-Weighted AUROC Aggregation (FVWA) — *aggregation novelty*

**The problem it solves:** Standard FedAvg aggregates client updates weighted solely by sample count. In a multi-hospital psychiatric setting, hospitals are highly non-IID — they differ in patient demographics, medication protocols, class imbalance ratios, and positive event rates. A hospital whose local model fits poorly (low AUROC) contributes an update that actively hurts the global model's ability to detect rare dangerous events. Weighting purely by sample size gives such hospitals disproportionate influence simply because they happen to be large.

**What FVWA does:** Replaces the sample-count weight with `w_k = N_k × AUROC_k` — each hospital's contribution is scaled by both its dataset size (volume) and its local validation AUROC on the dangerous-action prediction task (quality). Hospitals that discriminate well between safe and at-risk patients are up-weighted; those with poor local signal are down-weighted. The aggregation remains a single weighted average with no additional communication rounds or messages.

**Why AUROC specifically — not accuracy or loss:** Dangerous-action events are rare (severe class imbalance). Accuracy is misleading under imbalance — a hospital that predicts "never at risk" for every patient achieves high accuracy but zero clinical utility. Loss values are scale-dependent and not directly comparable across hospitals with different class ratios. AUROC, by contrast, measures the model's ability to *rank* at-risk patients above safe ones regardless of class distribution, making it the only metric that directly reflects clinical discrimination quality in an imbalanced, non-IID multi-hospital setting.

**Why the existing literature has not explored this** *(contextual rationale — not explicitly stated in the paper)*: Most federated learning aggregation research operates on benchmark datasets (CIFAR, FEMNIST, Shakespeare) where class balance is mild and all clients share the same task objective. In those settings, sample-count weighting is a reasonable proxy for client quality. The psychiatric clinical monitoring setting is categorically different: positive event rates vary dramatically across hospitals (some sites may have fewer than 5% positive samples), the task is inherently imbalanced, and a single poorly-calibrated hospital can substantially degrade global model performance. The field has not recognised AUROC as an aggregation signal because prior federated clinical work either (a) uses balanced datasets, (b) reports only accuracy-based metrics, or (c) treats all clients as equal contributors by design. FVWA is the first formulation to use discrimination quality — measured by AUROC — as an explicit weighting criterion in federated aggregation for clinical risk prediction.

**Why it is novel:** AUROC-weighted federated aggregation has not previously been applied to psychiatric inpatient risk prediction or to any federated clinical monitoring setting.

---

## Repository Structure

```
Codes/
├── model_fedtft.py              # Shared-head TFT model (R1–R3 ablations)
├── model_fedtft_hdfp.py         # Horizon-decoupled TFT model (R4 / FedTFT)
├── fvwa.py                      # FVWA aggregation module (imported by both servers)
├── fl_client_fedtft.py          # FL client — FedTFT (R4)
├── fl_server_fedtft.py          # FL server — FedTFT (R4)
├── fl_client_ablation.py        # FL client — ablation rows R1–R3
├── fl_server_ablation.py        # FL server — ablation rows R1–R3 (toggleable FVWA / mu)
├── fl_client_baseline.py        # FL client — baseline models
├── fl_server_baseline.py        # FL server — baseline models
├── train_centralized.py         # Centralized training (privacy upper-bound reference)
├── latency_benchmark.py         # Inference latency benchmark (Table IV)
├── run_all_experiments.sh       # Master runner: all ablation + baseline experiments
├── run_analysis_pipeline.sh     # Sequential analysis pipeline
│
├── preprocessing/
│   ├── preprocessing_final.ipynb       # Finalized preprocessing pipeline (baseline)
│   └── sensitivity_preprocessing.ipynb # Preprocessing sensitivity variants (Table V)
│
├── baselines/                   # Baseline model implementations
│   ├── FEDformer.py
│   ├── iTransformer.py
│   ├── PatchTST.py
│   └── DLinear.py
│
├── experiments/
│   ├── ablation/
│   │   ├── run_ablation.sh              # Run R1–R4 ablation chain
│   │   ├── run_baselines.sh             # Run all baseline comparisons
│   │   ├── run_fedtft_hdfp.sh           # Run FedTFT (R4) standalone
│   │   ├── run_sensitivity_fedtft.sh    # Run sensitivity variants
│   │   └── pvalue_ablation_fedtft.py    # Wilcoxon / paired t-test across ablation rows
│   │
│   ├── analysis/
│   │   ├── shap_analysis_fedtft.py      # Per-horizon SHAP feature importance (Figure 4)
│   │   ├── generate_figure4.py          # Render Figure 4 from SHAP JSON
│   │   ├── calibration_analysis.py      # Brier score + reliability diagrams
│   │   ├── significance_test.py         # Bootstrap 95% CI + McNemar's test
│   │   ├── compute_ablation_ci.py       # Bootstrap CI for ablation table AUROC
│   │   ├── feature_ablation.py          # Feature group ablation (cosinor / location / treatment)
│   │   ├── sensitivity_analysis.py      # Sensitivity to seq_len / hidden_dim / dropout
│   │   ├── heterogeneity_analysis.py    # Robustness under data heterogeneity
│   │   └── missingness_robustness.py    # Performance under missing data / client dropout
│   │
│   └── preprocessing/
│       ├── preprocessing_ablation.py    # Preprocessing variant ablation (P0–P7)
│       └── run_preprocessing_ablation.sh
```

---

## Data

The raw dataset (SAFER wearable study) is not publicly available due to patient privacy and IRB restrictions. The preprocessing notebooks include comments describing each processing step and the rationale for all feature selection decisions.

**Input format (after preprocessing):** Memory-mapped `.npy` arrays per hospital split:
```
last_npy_data/
├── GlobalData/
│   ├── static_data.npy       # [N, 14]  static features
│   ├── sequence_data.npy     # [N, 192, 25]  sequential features (48h @ 15-min)
│   └── targets.npy           # [N, 3]   binary labels (1h / 1d / 1w)
└── HospitalsData/
    └── <hospital_id>/
        ├── static_train.npy / static_val.npy / static_test.npy
        ├── sequence_train.npy / sequence_val.npy / sequence_test.npy
        └── targets_train.npy / targets_val.npy / targets_test.npy
```
---

## Requirements

- Python 3.11
- PyTorch 1.13.1
- Flower (flwr) 1.5.0
- torchmetrics
- scikit-learn
- numpy
- pandas
- shap
- bayesian-optimization
- matplotlib, scipy, tqdm

Install dependencies:
```bash
conda create -n fedtft_env python=3.11
conda activate fedtft_env
pip install torch==1.13.1 flwr==1.5.0 torchmetrics scikit-learn numpy pandas shap bayesian-optimization matplotlib scipy tqdm
```

---

## Reproducing Results

### Step 1: Preprocessing

Open and run `preprocessing/preprocessing_final.ipynb`. Set `RAW_CSV` to your local data path.
- **Cell 1:** Sequence creation (cosinor features, augmentation, patient-level split)
- **Cell 2:** Memmap creation (converts pickles to `.npy` arrays)

For sensitivity analysis variants (Table V), use `preprocessing/sensitivity_preprocessing.ipynb`.

### Step 2: Run all FL experiments (Table III)

```bash
# All experiments, 3 seeds (run once per seed)
bash run_all_experiments.sh all 1
bash run_all_experiments.sh all 2
bash run_all_experiments.sh all 3
```

To run a single row:
```bash
bash run_all_experiments.sh R4_fedtft 1    # FedTFT (R4), seed 1
bash run_all_experiments.sh R1_fedavg  1   # FedAvg baseline (R1)
```

### Step 3: Centralized training (Table III, centralized reference)

```bash
python train_centralized.py \
  --data_root /path/to/last_npy_data \
  --seeds 1 2 3 \
  --output results/centralized/
```

### Step 4: Analysis pipeline

```bash
bash run_analysis_pipeline.sh
```

Runs in sequence: significance tests → missingness robustness → heterogeneity analysis → SHAP.

---

## Ablation Configurations (Table III)

| Row | Name | Aggregation | Proximal term | Output head |
|-----|------|-------------|---------------|-------------|
| R1  | FedAvg | Sample-count (FedAvg) | None (μ=0) | Shared (64→3) |
| R2  | FedProx | Sample-count (FedAvg) | FedProx μ=1e-5 | Shared (64→3) |
| R3  | +FVWA | FVWA (N_k × AUROC_k) | FedProx μ=1e-5 | Shared (64→3) |
| R4  | FedTFT | FVWA (N_k × AUROC_k) | FedProx μ=1e-5 | Decoupled (3×64→1) |

Run individual ablation rows:
```bash
# R1 — FedAvg
bash experiments/ablation/run_ablation.sh R1_fedavg

# R4 — FedTFT (full model)
bash experiments/ablation/run_fedtft_hdfp.sh
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 192 timesteps (48 h @ 15-min) |
| Static features | 14 |
| Sequential features | 25 |
| Hidden dimension | 64 |
| Peak learning rate | 3 × 10⁻⁵ (OneCycleLR) |
| Local epochs per round | 6 |
| FedProx μ | 1 × 10⁻⁵ |
| Batch size | 32 |

---

## FVWA Aggregation

FVWA is implemented in `fvwa.py` and imported by both `fl_server_fedtft.py` and `fl_server_ablation.py`.

```
w_k  = N_k × AUROC_k
w̄_k  = w_k / Σ_j w_j
θ*   = Σ_k w̄_k · θ_k
```

Falls back to standard FedAvg (sample-count only) if any client does not report `val_auroc`.

---

## Citation

If you use this code, please cite:

```
[Citation will be added upon acceptance]
```

The implementation code is publicly available at:
https://github.com/gachon-CClab/FedOps/tree/main/multimodal/Usecases/FedTFT

---

## License

For research use only. The model and code are released for reproducibility purposes in connection with the above manuscript. The raw clinical dataset is not included and is not available for public release.
