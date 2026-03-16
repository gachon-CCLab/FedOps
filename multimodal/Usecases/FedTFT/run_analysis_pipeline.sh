#!/bin/bash
# run_analysis_pipeline.sh — Sequential analysis pipeline for FedTFT paper
# Steps:
#   1. significance_test.py   (bootstrap CI for event-recall, event-F1, Brier + paired tests)
#   2. missingness_robustness.py
#   3. heterogeneity_analysis.py
#   4. shap_analysis_fedtft.py (Figure 3 + Figure 4)
#
# Prerequisites: run_all_experiments.sh (all 3 seeds) must be complete first.
# Run from FedTFT_paper/ root directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" results/significance results/shap

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# ── 1. significance_test ─────────────────────────────────────────────────────
log "=== Step 1: significance_test (bootstrap CIs + paired tests) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:baselines PYTHONUNBUFFERED=1 \
  $PYTHON -u experiments/analysis/significance_test.py \
    --checkpoint   patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth \
    --output       results/significance/ \
    2>&1 | tee "$LOG_DIR/significance_test.log"
log "=== Step 1 complete ==="

# ── 2. missingness_robustness ────────────────────────────────────────────────
log "=== Step 2: missingness_robustness (Table V Part B) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:baselines PYTHONUNBUFFERED=1 \
  $PYTHON -u experiments/analysis/missingness_robustness.py \
    --checkpoint patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth \
    --thresholds_json checkpoints/best_thresholds.json \
    --rates 0.0 0.1 0.2 0.3 \
    --output results/significance/missingness.json \
    2>&1 | tee "$LOG_DIR/missingness_robustness.log"
log "=== Step 2 complete ==="

# ── 3. heterogeneity_analysis ────────────────────────────────────────────────
log "=== Step 3: heterogeneity_analysis (Table V Part B — Dirichlet, participation) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:baselines PYTHONUNBUFFERED=1 \
  $PYTHON -u experiments/analysis/heterogeneity_analysis.py \
    --checkpoint patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth \
    --output results/significance/heterogeneity.json \
    2>&1 | tee "$LOG_DIR/heterogeneity_analysis.log"
log "=== Step 3 complete ==="

# ── 4. shap_analysis_fedtft ──────────────────────────────────────────────────
log "=== Step 4: shap_analysis_fedtft (Figure 3 + Figure 4, slow ~2-4h) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:baselines PYTHONUNBUFFERED=1 \
  $PYTHON -u experiments/analysis/shap_analysis_fedtft.py \
    --checkpoint patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth \
    --model_variant fedtft \
    --hospital all \
    --n_background 100 \
    --n_explain 200 \
    --output results/shap/ \
    2>&1 | tee "$LOG_DIR/shap_analysis.log"
log "=== Step 4 complete ==="

log "======================================================="
log "Analysis pipeline complete."
log "  Significance:    results/significance/"
log "  Missingness:     results/significance/missingness.json"
log "  Heterogeneity:   results/significance/heterogeneity.json"
log "  SHAP:            results/shap/"
log "======================================================="
