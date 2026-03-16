#!/bin/bash
# run_baselines.sh — Launch baseline FL experiments (TimesNet, DLinear, etc.)
# Usage: bash experiments/run_baselines.sh [MODEL]
#   MODEL: timesnet | dlinear | patchtst | itransformer | fedformer (default: timesnet)
# Run from FedTFT_paper/ root directory

MODEL="${1:-timesnet}"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Baseline FL experiment: $MODEL ==="
echo "Client: fl_client_baseline.py, Server: fl_server_baseline.py"
echo ""

# Edit fl_client_baseline.py and fl_server_baseline.py to activate the desired model
# before running this script. The import line to activate is at line 18.

python fl_server_baseline.py &
SERVER_PID=$!
echo "Server started (PID=$SERVER_PID)"
sleep 5

python fl_client_baseline.py 0 &
python fl_client_baseline.py 1 &
python fl_client_baseline.py 2 &
echo "All 3 clients started"

wait
echo ""
echo "=== Baseline ($MODEL) experiment complete ==="
