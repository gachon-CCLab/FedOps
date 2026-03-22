#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Master experiment runner for FedTFT paper
#
# Strategy:
#   GPU0 port 8089 — Ablation chain  R1→R2→R3→R4→C1→C2  (sequential)
#   GPU1 port 8190 — Baselines       FEDFormer→iTransformer→PatchTST→DLinear (sequential)
#   Both groups run in parallel; each group serialises to avoid port conflicts.
#

#
# Logs: logs/<name>_seed<N>_<timestamp>.log  (stdout+stderr, server+all clients)
#
# Usage:
#   bash run_all_experiments.sh               # seed=1, all experiments
#   bash run_all_experiments.sh all 2         # seed=2  (separate result dirs, safe re-run)
#   bash run_all_experiments.sh ablation      # only GPU0 ablation chain, seed=1
#   bash run_all_experiments.sh baselines 2   # only GPU1 baselines, seed=2
#   bash run_all_experiments.sh R3_fvwa       # single named experiment, seed=1
#   bash run_all_experiments.sh R3_fvwa 3     # single experiment, seed=3
#
# For 3-seed paper mean, run:
#   bash run_all_experiments.sh all 1
#   bash run_all_experiments.sh all 2
#   bash run_all_experiments.sh all 3
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" results/ablation results/baselines results/sensitivity

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILTER="${1:-all}"
SEED="${2:-1}"

PORT_GPU0=8089
PORT_GPU1=8190

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# =============================================================================
# Core runner:
#   run_exp NAME GPU PORT SERVER_SCRIPT SERVER_FLAGS CLIENT_SCRIPT CLIENT_FLAGS
#
# All Python servers accept --seed $SEED → result saved to:
#   results/<group>/<name>/seed<N>/result.json   (never overwrites previous seeds)
# =============================================================================
run_exp() {
    local NAME="$1"
    local GPU="$2"
    local PORT="$3"
    local SERVER_SCRIPT="$4"
    local SERVER_FLAGS="$5"
    local CLIENT_SCRIPT="$6"
    local CLIENT_FLAGS="$7"
    local LOGFILE="$LOG_DIR/${NAME}_seed${SEED}_${TIMESTAMP}.log"

    # Skip if a specific name was requested and this isn't it
    if [[ "$FILTER" != "all" && "$FILTER" != "ablation" && "$FILTER" != "baselines" ]]; then
        [[ "$NAME" != "$FILTER" ]] && return 0
    fi

    log ">>> START  $NAME   GPU=$GPU  port=$PORT  seed=$SEED"
    log "    Log  → $LOGFILE"
    log "    Results will be in: results/*/seed${SEED}/"

    # Kill any leftover on this port
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2

    # Clear persisted FedTFT horizon heads for a clean run
    if [[ "$CLIENT_SCRIPT" == "fl_client_fedtft.py" ]]; then
        for hospital in "동국대" "서울대병원" "용인관리자"; do
            local H="patient_level_split/last_npy_data/HospitalsData/${hospital}/horizon_heads_fedtft.pth"
            [[ -f "$H" ]] && rm "$H" && log "    Cleared $H"
        done
    fi

    {
        echo "================================================"
        echo "EXPERIMENT : $NAME"
        echo "SEED       : $SEED"
        echo "GPU        : $GPU   PORT: $PORT"
        echo "SERVER     : python $SERVER_SCRIPT $SERVER_FLAGS --port $PORT --seed $SEED"
        echo "CLIENT     : python $CLIENT_SCRIPT [0|1|2] $CLIENT_FLAGS --port $PORT"
        echo "START      : $(date)"
        echo "================================================"
    } > "$LOGFILE"

    # Launch server
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SERVER_SCRIPT" $SERVER_FLAGS \
        --port "$PORT" --seed "$SEED" \
        >> "$LOGFILE" 2>&1 &
    local SERVER_PID=$!

    sleep 6  # let server bind

    # Launch 3 hospital clients
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$CLIENT_SCRIPT" 0 $CLIENT_FLAGS \
        --port "$PORT" >> "$LOGFILE" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$CLIENT_SCRIPT" 1 $CLIENT_FLAGS \
        --port "$PORT" >> "$LOGFILE" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$CLIENT_SCRIPT" 2 $CLIENT_FLAGS \
        --port "$PORT" >> "$LOGFILE" 2>&1 &

    wait $SERVER_PID 2>/dev/null || wait

    echo "DONE: $NAME seed=$SEED  $(date)" >> "$LOGFILE"
    log "    DONE   $NAME  → results/*/seed${SEED}/result.json"
    sleep 3
}

# =============================================================================
# GPU0 group: Ablation chain R1→R4, C1, C2
# =============================================================================
run_ablation_chain() {
    [[ "$FILTER" == "baselines" ]] && return 0
    log "=== GPU0: Ablation chain starting (seed=$SEED) ==="

    # R1 — FedAvg: no proximal, uniform avg, shared output head
    run_exp "R1_fedavg" 0 $PORT_GPU0 \
        "fl_server_ablation.py" "--ablation_name R1_fedavg --proximal_mu 0.0" \
        "fl_client_ablation.py" "--proximal_mu 0.0"

    # R2 — +FedProx (fixed mu)
    run_exp "R2_fedprox" 0 $PORT_GPU0 \
        "fl_server_ablation.py" "--ablation_name R2_fedprox --proximal_mu 1e-5" \
        "fl_client_ablation.py" "--proximal_mu 1e-5"

    # R3 — +FVWA accuracy-weighted aggregation
    run_exp "R3_fvwa" 0 $PORT_GPU0 \
        "fl_server_ablation.py" "--ablation_name R3_fvwa --proximal_mu 1e-5 --use_fvwa" \
        "fl_client_ablation.py" "--proximal_mu 1e-5"

    # R4 — FedTFT full: decoupled heads + FVWA
    run_exp "R4_fedtft" 0 $PORT_GPU0 \
        "fl_server_fedtft.py" "--result_tag R4_fedtft" \
        "fl_client_fedtft.py" ""

    # Row A — FedAvg + decoupled heads (A/B control)
    run_exp "A_fedavg_decoupled" 0 $PORT_GPU0 \
        "fl_server_fedtft.py" "--result_tag A_fedavg_decoupled --use_fedavg" \
        "fl_client_fedtft.py" "--proximal_mu 0.0"

    log "=== GPU0: Ablation chain complete. Results in results/ablation/*/seed${SEED}/ ==="
}

# =============================================================================
# GPU1 group: Baselines
# =============================================================================
run_baselines() {
    [[ "$FILTER" == "ablation" ]] && return 0
    log "=== GPU1: Baselines starting (seed=$SEED) ==="

    for MODEL in fedformer itransformer patchtst dlinear; do
        run_exp "baseline_${MODEL}" 1 $PORT_GPU1 \
            "fl_server_baseline.py" "--model_name ${MODEL}" \
            "fl_client_baseline.py" "--model_name ${MODEL}"
    done

    log "=== GPU1: Baselines complete. Results in results/baselines/*/seed${SEED}/ ==="
}

# =============================================================================
# Dispatch
# =============================================================================
log "========================================================"
log "FedTFT Master Experiment Runner"
log "Filter : $FILTER    Seed : $SEED"
log "Timestamp : $TIMESTAMP"
log "Log dir   : $LOG_DIR"
log ""
log "Result layout (no overwrites guaranteed):"
log "  results/ablation/<name>/seed${SEED}/result.json"
log "  results/baselines/<model>/seed${SEED}/result.json"
log ""
log "Monitor:  tail -f $LOG_DIR/*_seed${SEED}_${TIMESTAMP}.log"
log "========================================================"

case "$FILTER" in
    all)
        # Both GPU groups run simultaneously
        run_ablation_chain &
        GPU0_BG=$!
        run_baselines &
        GPU1_BG=$!
        log "GPU0 PID=$GPU0_BG (ablation)  |  GPU1 PID=$GPU1_BG (baselines)"
        wait $GPU0_BG
        wait $GPU1_BG
        ;;
    ablation)
        run_ablation_chain
        ;;
    baselines)
        run_baselines
        ;;
    baseline_*)
        MODEL="${FILTER#baseline_}"
        run_exp "$FILTER" 1 $PORT_GPU1 \
            "fl_server_baseline.py" "--model_name ${MODEL}" \
            "fl_client_baseline.py" "--model_name ${MODEL}"
        ;;
    *)
        # Single ablation config name (e.g. R3_fvwa)
        run_ablation_chain
        ;;
esac

log "========================================================"
log "All experiments finished (seed=$SEED)."
log ""
log "Post-processing (run once all 3 seeds are done):"
log "  conda activate fedmap_env && cd $SCRIPT_DIR"
log "  python experiments/analysis/significance_test.py     # CIs + p-values"
log "  python experiments/analysis/feature_ablation.py      # w/o aug/cosinor/location"
log "  python experiments/analysis/missingness_robustness.py"
log "  python experiments/analysis/heterogeneity_analysis.py"
log "  python experiments/analysis/shap_analysis_fedtft.py   # Figure 3 + Figure 4"
log "========================================================"
