"""FedOps LMECS Selection strategy using Flower"""
from typing import List, Tuple, Union, Optional, Dict

from flwr.common import Metrics, Scalar
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from logging import DEBUG, INFO

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate with weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    
    # total num_examples
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)

    # Metrics key like acc, auc etc.
    keys = set(k for _, metric in metrics for k in metric.keys())

    # The weighted average metric.
    weighted_performances = {}
    for key in keys:
        performances = [num_examples * float(metric[key]) for num_examples, metric in metrics if key in metric]

        weighted_performance = sum(performances) / total_examples if total_examples > 0 else 0

        weighted_performances[key] = weighted_performance

    return {**weighted_performances, "client_performances": metrics}


def get_top_clients_by_metrics(history, top_ratio, metric, standard):
    """
    Get top clients based on the specified metric (loss or accuracy) and ratio.
    
    :param history: History object containing clients' performances.
    :param top_ratio: Float indicating the ratio of top clients to retrieve.
    :param metric: String indicating which metric to use for ranking ('loss' or 'accuracy').
    :return: List of dictionaries with 'cid', 'loss', and 'accuracy' of the top clients.
    """
    # Assuming that the latest round is the last item in 'client_performances'
    latest_round_data = history.metrics_distributed['client_performances'][-1][1]
    
    # Sort the clients based on the specified metric
    sorted_clients = sorted(latest_round_data, key=lambda x: x[1][metric], reverse=True)
    
    
    if standard == 'max':
        # Determine the number of top clients to retrieve
        num_top_clients = max(int(len(sorted_clients) * top_ratio), 1)
    elif standard == 'min':
        num_top_clients = min(int(len(sorted_clients) * top_ratio), 1)
    
    # Retrieve the top clients based on the metric
    top_clients = sorted_clients[:num_top_clients]
    
    # Format the data to return a list of dictionaries
    top_clients_data = [{'cid': client[1]['cid'], 'loss': client[1]['loss'], 'accuracy': client[1]['accuracy'], metric: client[1][metric]} 
                        for client in top_clients]
    
    # Extract only the cids
    top_clients_cids = [str(client[1]['cid']) for client in top_clients]
    
    return top_clients_cids, top_clients_data
    

class CustomSelection(FedAvg):
    def __init__(self, selection_use, selection_ratio, metric, standard, **kwargs):
        super().__init__(**kwargs)
        self.selection_use = selection_use
        self.selection_ratio = selection_ratio
        self.metric = metric
        self.standard = standard
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager, history: History
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        client_select_ids = None
        if server_round > 1:
            if self.selection_use:
                # top_ratio 및 metric 지정 필요
                client_select_ids, client_select_data = get_top_clients_by_metrics(history, self.selection_ratio, self.metric, self.standard)
                log(INFO, f"Round: {server_round} Selected Clients Data")
                log(INFO, f"{client_select_data}")
            else: 
                pass
        
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, client_select_ids=client_select_ids
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]