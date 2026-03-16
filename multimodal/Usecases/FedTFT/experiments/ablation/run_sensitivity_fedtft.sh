#!/bin/bash
# run_sensitivity_fedtft.sh — Run FedTFT sensitivity experiments (preprocessing hyperparams)
#
# Each variant reuses fl_server_fedtft.py / fl_client_fedtft.py with --data_root
# pointing to the variant's preprocessed last_npy_data/ folder.
#
# Prerequisites:
#   1. Run preprocessing/sensitivity_preprocessing.ipynb first to generate the
#      variant-specific last_npy_data/ folders under:
#        patient_level_split/sensitivity/<variant>/last_npy_data/
#
# Usage:
#   bash experiments/ablation/run_sensitivity_fedtft.sh            # seed=1, all variants
#   bash experiments/ablation/run_sensitivity_fedtft.sh "" 2       # seed=2, all variants
#   bash experiments/ablation/run_sensitivity_fedtft.sh aug_sigma025     # single variant, seed=1
#   bash experiments/ablation/run_sensitivity_fedtft.sh aug_sigma025 2   # single variant, seed=2
#
# Results: results/sensitivity/<variant>/seed<N>/result.json  (never overwrites)
# Run from FedTFT_paper/ root directory.

# ── cd to FedTFT_paper root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PORT=8089
FILTER="${1:-}"
SEED="${2:-1}"

# ── Helper: run one FL experiment for a sensitivity variant ───────────────────
run_sensitivity() {
    local VARIANT="$1"   # e.g. aug_sigma025
    local SEQ_LEN="$2"   # e.g. 192  (96 or 288 for seq-length variants)
    local DATA_ROOT="patient_level_split/sensitivity/${VARIANT}/last_npy_data"
    local HEADS_DIR="${DATA_ROOT}/HospitalsData"

    echo ""
    echo "======================================================================="
    echo "=== SENSITIVITY: ${VARIANT}  (seq_len=${SEQ_LEN}) ==="
    echo "======================================================================="

    # Abort if preprocessing hasn't been run yet
    if [ ! -d "${DATA_ROOT}/GlobalData" ]; then
        echo "[SKIP] ${DATA_ROOT}/GlobalData not found."
        echo "       Run sensitivity_preprocessing.ipynb first for tag '${VARIANT}'."
        return 1
    fi

    # Kill any leftover processes on this port
    fuser -k ${PORT}/tcp 2>/dev/null; sleep 2

    # Clear saved horizon heads so each variant starts fresh
    for hospital in "동국대" "서울대병원" "용인관리자"; do
        HEADS="${HEADS_DIR}/${hospital}/horizon_heads_fedtft.pth"
        [ -f "$HEADS" ] && rm "$HEADS" && echo "  Cleared heads: $HEADS"
    done

    # Launch server  (--seed isolates this run: results/sensitivity/<variant>/seed<N>/result.json)
    python fl_server_fedtft.py \
        --data_root  "${DATA_ROOT}" \
        --seq_len    "${SEQ_LEN}" \
        --result_tag "${VARIANT}" \
        --seed       "${SEED}" &
    SERVER_PID=$!
    sleep 5

    # Launch 3 hospital clients
    python fl_client_fedtft.py 0 --data_root "${DATA_ROOT}" --seq_len "${SEQ_LEN}" &
    python fl_client_fedtft.py 1 --data_root "${DATA_ROOT}" --seq_len "${SEQ_LEN}" &
    python fl_client_fedtft.py 2 --data_root "${DATA_ROOT}" --seq_len "${SEQ_LEN}" &

    wait $SERVER_PID 2>/dev/null || wait
    echo "=== ${VARIANT} done. Results → results/sensitivity/${VARIANT}/seed${SEED}/result.json ==="
    sleep 3
}

# ── Variant table ─────────────────────────────────────────────────────────────
#   variant tag          seq_len
VARIANTS=(
    "aug_sigma025   192"
    "aug_sigma100   192"
    "cosinor_24_120 192"
    "cosinor_72_192 192"
    "seqlen_096      96"
    "seqlen_288     288"
)

# ── Run variants ──────────────────────────────────────────────────────────────
for entry in "${VARIANTS[@]}"; do
    VTAG=$(echo "$entry" | awk '{print $1}')
    VLEN=$(echo "$entry" | awk '{print $2}')
    if [ -n "$FILTER" ] && [ "$VTAG" != "$FILTER" ]; then
        continue
    fi
    run_sensitivity "$VTAG" "$VLEN"
done

echo ""
echo "======================================================================="
echo "All requested sensitivity experiments complete (seed=${SEED})."
echo ""
echo "Results are in results/sensitivity/<variant>/seed${SEED}/result.json"
echo ""
echo "For 3-seed mean, run:"
echo "  bash experiments/ablation/run_sensitivity_fedtft.sh  \"\" 1"
echo "  bash experiments/ablation/run_sensitivity_fedtft.sh  \"\" 2"
echo "  bash experiments/ablation/run_sensitivity_fedtft.sh  \"\" 3"
echo "======================================================================="
