"""FedOps Client using Flower"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for DNN training."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_train,
        client_test,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        cid: int
    ):  # pylint: disable=too-many-arguments
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_train = client_train
        self.client_test = client_test
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.cid = cid
        self.curr_round = 0
        
        # wandb.login(key='eb56a89a7457704c55789c584a3546c707d737f8')
        # task = 'Only Local Model - FedAvg 100'
        # name = f'{self.cid} client'
        # self.run = wandb.init(project=task, name=name)

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using
        BN layers."""
        self.model.train()
        # pylint: disable = no-else-return
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn
        layer if available."""
        self.model.train()
        # pylint: disable=not-callable
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
        

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)

        num_epochs = self.num_epochs
            
        self.curr_round = int(config["curr_round"])
        
        loss, accuracy = self.client_train(
            self.model,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
        )

        return self.get_parameters({}), len(self.trainloader), {"loss": loss, "accuracy": accuracy}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy, metrics = self.client_test(self.model, self.valloader, self.device, cid = self.cid)
        if metrics!=None:
            return float(loss), len(self.valloader), {"cid": int(self.cid), "loss": float(loss), "accuracy": float(accuracy), **metrics}
        else:
            return float(loss), len(self.valloader), {"cid": int(self.cid), "loss": float(loss), "accuracy": float(accuracy)}

def gen_client_fn(
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    client_train,
    client_test,
    learning_rate: float,
    model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments


    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            net,
            trainloader,
            valloader,
            client_train,
            client_test,
            device,
            num_epochs,
            learning_rate,
            cid,
        )

    return client_fn
