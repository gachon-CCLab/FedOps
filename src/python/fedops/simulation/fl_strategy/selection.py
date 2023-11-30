import random
from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class ClientSelectionManager(SimpleClientManager):
    """In the fit phase, clients must be sampled from the training client list.

    And in the evaluate stage, clients must be sampled from the validation client list.
    So we modify 'fedmeta_client_manager' to sample clients from [cid: List] for each
    list.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        client_select_ids: Optional[List[str]] = None,  # 추가된 매개변수
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        
        # If specific client IDs are provided, use them for sampling
        if client_select_ids is not None:
            available_cids = client_select_ids
            return [self.clients[cid] for cid in available_cids]
        else: 
            sampled_cids = random.sample(available_cids, num_clients)
            return [self.clients[cid] for cid in sampled_cids]

        # if num_clients > len(available_cids):
        #     log(
        #         INFO,
        #         "Sampling failed: number of available clients"
        #         " (%s) is less than number of requested clients (%s).",
        #         len(available_cids),
        #         num_clients,
        #     )
        #     return []

        # sampled_cids = random.sample(available_cids, num_clients)
        