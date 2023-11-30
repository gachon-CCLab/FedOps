
from typing import Dict, Tuple, cast
from logging import WARNING
import flwr
from flwr.common import Parameters, Scalar, NDArrays, NDArray
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate
import numpy as np
from io import BytesIO


class MobileStrategy(flwr.server.strategy.FedAvg):
    
    def __init__(self, client_device, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the initializer of the parent class (FedAvg)
        self.client_device = client_device
   
    
    def ndarrays_to_parameters(self, ndarrays: NDArrays) -> Parameters:
        """Convert NumPy ndarrays to parameters object."""
        tensors = [self.ndarray_to_bytes(ndarray) for ndarray in ndarrays]

        if self.client_device == 'android':
            tensor_type = "numpy.nda"
        else:
            tensor_type = "numpy.ndarray"
            
        return Parameters(tensors=tensors, tensor_type=tensor_type)

    def parameters_to_ndarrays(self, parameters: Parameters) -> NDArrays:
        """Convert parameters object to NumPy weights."""
        return [self.bytes_to_ndarray(tensor) for tensor in parameters.tensors]

    def ndarray_to_bytes(self, ndarray: NDArray) -> bytes:
        """Serialize NumPy array to bytes."""
        if self.client_device == 'android':
            return ndarray.tobytes()
        else:
            bytes_io = BytesIO()
            np.save(bytes_io, ndarray, allow_pickle=False)  # type: ignore
            return bytes_io.getvalue()
        
    def bytes_to_ndarray(self, tensor: bytes) -> NDArray:
        """Deserialize NumPy array from bytes."""
        if self.client_device == 'android':
            ndarray_deserialized = np.frombuffer(tensor, dtype=np.float32)  # type: ignore
            return cast(NDArray, ndarray_deserialized)
        else:
            bytes_io = BytesIO(tensor)
            ndarray_deserialized = np.load(bytes_io, allow_pickle=False)  # type: ignore
            return cast(NDArray, ndarray_deserialized)
        

    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (self.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = self.ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # if parameters_aggregated is not None:
        #     # Save weights
        #     print(f"Saving round {rnd} weights...")
            # np.savez(f"round-{rnd}-weights.npz", *parameters_aggregated)
        
        return parameters_aggregated, metrics_aggregated
        
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, Scalar]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = self.parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics