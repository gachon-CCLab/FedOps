"""
fvwa.py — Federated Volume-Weighted AUROC Aggregation (FVWA)

One of the two core methodological contributions of FedTFT (Section II.C).

FVWA replaces standard FedAvg on the server side while retaining FedProx
as the local client-side optimisation objective.

Aggregation rule:
    w_k  = N_k × auroc_k
    w̄_k  = w_k / Σ_j w_j
    θ*   = Σ_k w̄_k · θ_k

where N_k is the number of training samples at client k and auroc_k is the
macro-averaged validation AUROC reported by client k after local training.

Rationale:
  - N_k weighting (from FedAvg) accounts for dataset size heterogeneity.
  - auroc_k weighting up-weights clients whose local model generalises well
    on their validation set, discounting clients with poor local signal
    (e.g., very few positive events, severe class imbalance).
  - Falls back to plain sample-count weighting (FedAvg) when any client
    does not report val_auroc (e.g., R1/R2 ablation rows).

Usage (imported by fl_server_fedtft.py and fl_server_ablation.py):

    from fvwa import fvwa_aggregate

    aggregated_ndarrays = fvwa_aggregate(results)
"""

import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


def fvwa_aggregate(results):
    """
    Compute FVWA-weighted parameter average from a list of Flower fit results.

    Parameters
    ----------
    results : list of (ClientProxy, FitRes)
        Flower fit results from aggregate_fit().

    Returns
    -------
    aggregated : list of np.ndarray
        Weighted-averaged parameters as ndarrays (same order as model state_dict).
    norm_weights : list of float
        Normalised per-client weights (sums to 1.0), for logging.
    used_fvwa : bool
        True  → AUROC-weighted aggregation was applied.
        False → fell back to sample-count weighting (FedAvg).
    """
    val_aurocs = [fit_res.metrics.get("val_auroc", None) for _, fit_res in results]

    if all(v is not None for v in val_aurocs):
        # FVWA: N_k × auroc_k
        weights  = [fit_res.num_examples * fit_res.metrics["val_auroc"]
                    for _, fit_res in results]
        used_fvwa = True
    else:
        # Fallback: plain FedAvg sample-count weights
        weights   = [fit_res.num_examples for _, fit_res in results]
        used_fvwa = False

    total_w    = sum(weights) or 1.0
    norm_w     = [w / total_w for w in weights]

    param_lists = [parameters_to_ndarrays(fit_res.parameters)
                   for _, fit_res in results]
    aggregated  = [
        sum(param_lists[j][layer] * norm_w[j] for j in range(len(norm_w)))
        for layer in range(len(param_lists[0]))
    ]

    return aggregated, norm_w, used_fvwa
