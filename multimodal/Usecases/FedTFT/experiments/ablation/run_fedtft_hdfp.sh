#!/bin/bash
# run_fedtft_hdfp.sh — Launch FedTFT full experiment (R4)
# FedTFT = FedProx (mu=1e-5) + FVWA aggregation + 3 horizon-decoupled linear heads (all fully federated)
# All parameters (backbone + 3 heads) are aggregated via FVWA each round.
# Usage: bash experiments/ablation/run_fedtft_hdfp.sh
# Run from FedTFT_paper/ root directory

trap 'kill $(jobs -p) 2>/dev/null' EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== FedTFT (R4): Horizon-Decoupled Fully Federated Learning ==="
echo "Server: fl_server_fedtft.py (FVWA aggregation of backbone + all 3 horizon heads)"
echo "Clients: fl_client_fedtft.py (trains backbone + 3 linear heads, all sent to server)"
echo ""

# Clear saved horizon heads for a fresh run (comment out to resume from checkpoint)
for hospital in "동국대" "서울대병원" "용인관리자"; do
    HEADS="patient_level_split/last_npy_data/HospitalsData/$hospital/horizon_heads_fedtft.pth"
    if [ -f "$HEADS" ]; then
        echo "  Clearing old heads: $HEADS"
        rm "$HEADS"
    fi
done

python fl_server_fedtft.py --result_tag R4_fedtft &
SERVER_PID=$!
echo "Server started (PID=$SERVER_PID)"
sleep 5

python fl_client_fedtft.py 0 &   # 동국대
python fl_client_fedtft.py 1 &   # 서울대병원
python fl_client_fedtft.py 2 &   # 용인관리자
echo "All 3 clients started"

wait
echo ""
echo "=== FedTFT (R4) experiment complete ==="
echo "Best model saved to patient_level_split/last_npy_data/GlobalData/fedtft_backbone_best.pth"
echo "Per-round metrics saved to results/ablation/R4_fedtft/result.json"
