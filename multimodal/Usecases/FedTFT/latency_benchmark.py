import sys
import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "baselines"))

import time
import torch
import numpy as np

from baselines.FEDformer import FedformerClassifier
from baselines.iTransformer import ITransformerClassifier
from baselines.PatchTST import PatchTSTClassifier
from baselines.DLinear import DLinearClassifier
from model_fedtft_hdfp import TFTPredictor_FedTFT

MODELS = {
    "FEDFormer":    FedformerClassifier(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                                        d_model=128, n_heads=8, e_layers=4, d_ff=512,
                                        dropout=0.1, version='Fourier'),
    "iTransformer": ITransformerClassifier(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                                           d_model=128, n_heads=4, e_layers=4, d_ff=512,
                                           dropout=0.1, use_norm=True),
    "PatchTST":     PatchTSTClassifier(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                                       d_model=128, n_heads=4, e_layers=4, d_ff=512,
                                       dropout=0.1),
    "DLinear":      DLinearClassifier(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                                      kernel_size=25, individual=True, hidden_dim=256,
                                      dropout=0.1),
    "FedTFT":       TFTPredictor_FedTFT(input_dim=25, static_dim=14, hidden_dim=64),
}

WARMUP = 100
RUNS   = 300

results = {}

for name, model in MODELS.items():
    model.eval()
    model.cpu()

    static_x = torch.randn(1, 14)
    seq_x    = torch.randn(1, 192, 25)

    # --- batch-1 latency ---
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(static_x, seq_x)

    times = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(static_x, seq_x)
            times.append((time.perf_counter() - t0) * 1000)

    lat_b1 = float(np.median(times))

    # --- batch-32 amortized latency ---
    static_x32 = torch.randn(32, 14)
    seq_x32    = torch.randn(32, 192, 25)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(static_x32, seq_x32)

    times32 = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            _ = model(static_x32, seq_x32)
            times32.append((time.perf_counter() - t0) * 1000 / 32)

    lat_b32 = float(np.median(times32))
    results[name] = (lat_b1, lat_b32)
    print(f"{name:15s}  batch-1: {lat_b1:7.2f} ms/window   batch-32 amortized: {lat_b32:6.3f} ms/window")

print("\nSummary:")
for name, (b1, b32) in results.items():
    print(f"  {name}: {b1:.1f} ms (batch-1), {b32:.2f} ms (batch-32 amortized)")
