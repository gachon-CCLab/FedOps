"""Contains utility functions for CNN FL on MNIST."""

import pickle
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History


def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : str
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    # All client local models Performance(weighted_average)
    metric_type = "distributed"
    
    metric_dict = hist.metrics_distributed  # or hist.metrics_centralized for centralized metrics

    # Exclude 'cid' and 'client_performances' keys
    filtered_metric_dict = {k: v for k, v in metric_dict.items() if k not in ['cid', 'client_performances']}
    
    # Extract rounds for loss (assuming all metrics have the same number of rounds)
    rounds, _ = zip(*hist.losses_distributed)

    # Determine the number of metrics
    num_metrics = len(filtered_metric_dict)

    # Create subplots
    _, axs = plt.subplots(nrows=num_metrics, ncols=1, sharex="row", figsize=(8, 2*num_metrics))

    # Loop over each metric and plot
    for i, (metric_name, metric_values) in enumerate(filtered_metric_dict.items()):
        rounds, values = zip(*metric_values)
        axs[i].plot(np.asarray(rounds), np.asarray(values))
        axs[i].set_ylabel(metric_name.capitalize())

    # Set common labels and title
    plt.xlabel("Rounds")
    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")

    # Save plot
    plt.savefig(Path(save_plot_path) / Path(f"{suffix} {metric_type}_metrics.png"))
    plt.close()
    
    


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = None,
    default_filename: str = "results.pkl",
) -> None:
    """Save results from simulation to pickle.


    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name (so it doesn't.

        overwrite the file).
        """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history}
    if extra_results is not None:
        data = {**data, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
