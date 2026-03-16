#!/bin/bash
# run_ablation.sh — Run ablation configurations aligned with manuscript Table III (R1–R4, Row A).
#   R1: FedAvg (mu=0) + shared head
#   R2: FedProx (mu=1e-5) + FedAvg + shared head
#   R3: FedProx (mu=1e-5) + FVWA + shared head
#   R4: FedProx (mu=1e-5) + FVWA + decoupled heads = FedTFT (full model)
#    A: FedAvg (mu=0) + decoupled heads (controlled A/B comparison)
# Each config runs a full FL experiment (50 rounds, 3 clients).
# Results saved to results/ablation/<name>/result.json.
# Run from FedTFT_paper/ root directory.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PORT=8089

run_config() {
    local NAME="$1"
    local SERVER_FLAGS="$2"
    local CLIENT_FLAGS="$3"

    echo ""
    echo "================================================"
    echo "=== ABLATION: $NAME ==="
    echo "================================================"

    # Kill any leftover processes on this port
    fuser -k ${PORT}/tcp 2>/dev/null; sleep 2

    # Server
    python fl_server_ablation.py $SERVER_FLAGS &
    SERVER_PID=$!
    sleep 5

    # 3 clients
    python fl_client_ablation.py 0 $CLIENT_FLAGS &
    python fl_client_ablation.py 1 $CLIENT_FLAGS &
    python fl_client_ablation.py 2 $CLIENT_FLAGS &

    wait $SERVER_PID 2>/dev/null || wait
    echo "=== $NAME done. Results → results/ablation/$NAME/result.json ==="
    sleep 3
}

# ── R1: FedAvg (shared head) ────────────────────────────────────────────────────
run_config "R1_fedavg" \
    "--ablation_name R1_fedavg --proximal_mu 0.0" \
    "--proximal_mu 0.0"

# ── R2: FedProx + FedAvg (shared head) ─────────────────────────────────────────
run_config "R2_fedprox" \
    "--ablation_name R2_fedprox --proximal_mu 1e-5" \
    "--proximal_mu 1e-5"

# ── R3: FedProx + FVWA (shared head) ───────────────────────────────────────────
run_config "R3_fvwa" \
    "--ablation_name R3_fvwa --proximal_mu 1e-5 --use_fvwa" \
    "--proximal_mu 1e-5"

# ── R4: FedTFT (decoupled heads + FVWA) ────────────────────────────────────────
echo ""
echo "================================================"
echo "=== R4: FedTFT Full (decoupled heads + FVWA) ==="
echo "================================================"
fuser -k ${PORT}/tcp 2>/dev/null; sleep 2

# Clear saved horizon heads for a fresh run
for hospital in "동국대" "서울대병원" "용인관리자"; do
    HEADS="patient_level_split/last_npy_data/HospitalsData/$hospital/horizon_heads_fedtft.pth"
    [ -f "$HEADS" ] && rm "$HEADS"
done

python fl_server_fedtft.py --result_tag R4_fedtft &
SERVER_PID=$!
sleep 5

python fl_client_fedtft.py 0 &
python fl_client_fedtft.py 1 &
python fl_client_fedtft.py 2 &

wait $SERVER_PID 2>/dev/null || wait
echo "=== R4 done. Results → results/ablation/R4_fedtft/result.json ==="
sleep 3

# ── Row A: FedAvg + decoupled heads (A/B control) ──────────────────────────────
echo ""
echo "================================================"
echo "=== Row A: FedAvg + decoupled heads (A/B control) ==="
echo "================================================"
fuser -k ${PORT}/tcp 2>/dev/null; sleep 2

for hospital in "동국대" "서울대병원" "용인관리자"; do
    HEADS="patient_level_split/last_npy_data/HospitalsData/$hospital/horizon_heads_fedtft.pth"
    [ -f "$HEADS" ] && rm "$HEADS"
done

python fl_server_fedtft.py --result_tag A_fedavg_decoupled --use_fedavg &
SERVER_PID=$!
sleep 5

python fl_client_fedtft.py 0 --proximal_mu 0.0 &
python fl_client_fedtft.py 1 --proximal_mu 0.0 &
python fl_client_fedtft.py 2 --proximal_mu 0.0 &

wait $SERVER_PID 2>/dev/null || wait
echo "=== Row A done. Results → results/ablation/A_fedavg_decoupled/result.json ==="
sleep 3

echo ""
echo "========================================"
echo "All ablation runs complete (R1–R4, Row A)."
echo "Run: python experiments/ablation/pvalue_ablation_fedtft.py --seed 1"
echo "========================================"
