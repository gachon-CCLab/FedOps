from typing import Dict, Union

import flwr as fl
from hydra.utils import instantiate
from .fl_strategy import server
from .fl_strategy import client
from .fl_strategy.selection import ClientSelectionManager
import wandb

class FLSimulation:
    def __init__(self, cfg, trainloaders, valloaders, testloader, client_train, client_test, server_test, model) -> None:
        self.cfg = cfg,
        self.trainloaders = trainloaders,
        self.valloaders = valloaders,
        self.testloader = testloader,
        self.client_train = client_train
        self.client_test = client_test
        self.server_test = server_test
        self.model = model,
        
    def start(self,):
        # Because of getting Tuple format
        self.cfg = self.cfg[0]
        self.trainloaders = self.trainloaders[0]
        self.valloaders = self.valloaders[0]
        self.testloader = self.testloader[0]
        self.model = self.model[0]

        if self.cfg.wandb.use:
            wandb.login(key=self.cfg.wandb.key)
            task = self.cfg.wandb.project
            name = self.cfg.wandb.name
            run = wandb.init(project=task, name=name)
            wandb.config.update({
                "Epochs": self.cfg.num_epochs,
                "Rounds": self.cfg.num_rounds,
                "All Client Number": self.cfg.num_clients,
                "Selection Metric": self.cfg.strategy.metric,
                "Metric Standard": self.cfg.strategy.standard,
                "Seed": self.cfg.random_seed
            })
        else:
            run = None
            
        # prepare function that will be used to spawn each client
        client_fn = client.gen_client_fn(
            num_epochs= self.cfg.num_epochs,
            trainloaders=self.trainloaders,
            valloaders=self.valloaders,
            client_train=self.client_train,
            client_test=self.client_test,
            learning_rate=self.cfg.learning_rate,
            model=self.model,
        )

        # get function that will executed by the strategy's evaluate() method
        # Set server's device
        device = self.cfg.server_device
        evaluate_fn = server.gen_evaluate_fn(self.testloader, device=device, model=self.model, 
                                             wandb_use=self.cfg.wandb.use, run=run, server_test=self.server_test)

        # get a function that will be used to construct the config that the client's
        # fit() method will received        
        FitConfig = Dict[str, Union[bool, float]]

        def get_on_fit_config():
            def fit_config_fn(server_round: int):
                # resolve and convert to python dict
                fit_config: FitConfig = {}
                fit_config["curr_round"] = server_round  # add round info
                return fit_config

            return fit_config_fn

        # instantiate strategy according to config. Here we pass other arguments
        # that are only defined at run time.
        strategy = instantiate(
            self.cfg.strategy,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=get_on_fit_config(),
        )

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.cfg.num_rounds),
            client_resources={
                "num_cpus": self.cfg.client_resources.num_cpus,
                "num_gpus": self.cfg.client_resources.num_gpus,
            },
            server = server.CustomServer(strategy=strategy,client_manager=ClientSelectionManager()),
        )
        
        if self.cfg.wandb.use:
            run.finish()
        
        return history

        
        
