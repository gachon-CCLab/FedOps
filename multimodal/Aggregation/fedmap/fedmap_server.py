# fedmap/fedmap_server.py
import flwr as fl
from fedops.server.app import FLServer

class FedMAPServer(FLServer):
    """
    Drop-in replacement which lets us use a custom Flower Strategy.
    We don't modify FedOps; we only override .start() to plug the strategy.
    """
    def __init__(self, *, cfg, model, model_name, model_type, gl_val_loader, test_torch, strategy):
        super().__init__(cfg=cfg, model=model, model_name=model_name, model_type=model_type,
                         gl_val_loader=gl_val_loader, test_torch=test_torch)
        self.custom_strategy = strategy

    def start(self):
        # If custom strategy is set, run Flower server with it; else fall back
        if self.custom_strategy is not None:
            fl.server.start_server(
                server_address=self.cfg.server_address,
                config=fl.server.ServerConfig(num_rounds=self.cfg.num_rounds),
                strategy=self.custom_strategy,
            )
        else:
            super().start()
