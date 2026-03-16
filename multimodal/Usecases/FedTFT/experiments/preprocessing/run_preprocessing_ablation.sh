#!/bin/bash
# run_preprocessing_ablation.sh — Run preprocessing & augmentation ablation study
#
# Addresses Reviewer 2 (R2.5) and Reviewer 3 (R3.1, R3.2a):
#   "Preprocessing steps are complex; pipeline reproducibility unclear"
#   "Provide empirical justification for preprocessing hyperparameters"
#   "Augmentation vs no augmentation for positive-event sequences"
#
# Variants tested (P0–P7):
#   P0: No normalization
#   P1: StandardScaler (z-score) — DEFAULT
#   P2: MinMaxScaler
#   P3: RevIN (per-sample)
#   P4: StandardScaler + SMOTE
#   P5: StandardScaler + Jitter
#   P6: StandardScaler + Time masking
#   P7: StandardScaler + Mixup (α=0.2)
#
# Usage:
#   bash experiments/preprocessing/run_preprocessing_ablation.sh
#   bash experiments/preprocessing/run_preprocessing_ablation.sh --rounds 50
#
# Run from FedTFT_paper/ root directory.

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="results/preprocessing_ablation"
NUM_ROUNDS=30

# Parse optional --rounds argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)
            NUM_ROUNDS="$2"; shift 2 ;;
        --output)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "=== Preprocessing Ablation Study (P0–P7)       ==="
echo "=== Rounds: $NUM_ROUNDS                         ==="
echo "=== Output: $OUTPUT_DIR                         ==="
echo "=================================================="

python experiments/preprocessing/preprocessing_ablation.py \
    --num_rounds "$NUM_ROUNDS" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Preprocessing ablation complete ==="
    echo "Results → $OUTPUT_DIR/preprocessing_ablation_results.json"
    echo "Plots   → $OUTPUT_DIR/preprocessing_ablation_*.png"
else
    echo "[ERROR] Preprocessing ablation failed (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
