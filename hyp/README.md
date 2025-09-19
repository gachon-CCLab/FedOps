# ClusterOptunaFedAvg (Optional)

* **Client Profiling**
  Each round, the server records per-client validation signals (e.g., loss, accuracy, F1) and dataset size to maintain a lightweight profile of each client.

* **DBSCAN Clustering**
  Clients with similar performance/scale are automatically grouped via DBSCAN. Outliers (noise) are treated as single-client clusters to avoid distorting group behavior.

* **Cluster-Level HPO**
  For each cluster, the server runs Optuna to propose hyperparameters (learning rate, batch size, local epochs). The same proposal is applied to all clients in the cluster for that round, enabling stable, comparable evaluation.

* **Feedback via Cluster Averages**
  After training, the cluster’s score is computed as the mean of its members’ validation metrics. That averaged score is fed back to Optuna to update its search distribution for the next round.

* **Why it helps**
  Grouping similar clients turns noisy per-client signals into robust cluster signals, accelerating convergence and improving stability in non-IID settings.

# Central Evaluation & BestKeeper

* **Round-Wise Central Evaluation**
  After aggregation each round, the server evaluates the global model on a fixed central validation set and logs the results. A round snapshot of the global weights is also stored for auditing/regression checks.

* **Track the Best Checkpoint**
  A designated metric (default: accuracy; configurable, e.g., F1) is monitored throughout training. Whenever a new best score appears, the corresponding global parameters are kept as the current best.

* **Guaranteed Final Output**
  When training ends, the final exported model is **not** the last round by default—it is the **best-performing** checkpoint observed during training. This guards against late-round degradation due to overfitting or oscillations.

* **Benefits**
  Ensures the delivered model is the highest-quality version seen, improves reproducibility, and simplifies downstream evaluation and deployment.
