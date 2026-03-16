# FedTFT — Federated Temporal Fusion Transformer for Multi-Horizon Psychiatric Risk Prediction

---

## Overview

FedTFT is a federated learning framework for predicting imminent dangerous actions (1-hour, 1-day, 1-week horizons) from wearable sensor data collected across multiple hospitals, without sharing patient data.

Two core contributions:
- **Horizon-Decoupled Federated Prediction (HDFP):** replaces the single shared output head with three independent per-horizon heads, eliminating inter-horizon gradient interference during federation.
- **Federated Volume-Weighted AUROC Aggregation (FVWA):** server-side aggregation rule `w_k = N_k × AUROC_k` that up-weights clients whose local model generalises well, replacing standard FedAvg.

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

Results saved to `results/ablation/<name>/seed<N>/result.json` and `results/baselines/`.

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
| FL rounds | 50 (early stopping patience = 15) |
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
