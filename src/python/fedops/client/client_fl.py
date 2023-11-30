from collections import OrderedDict
import json, logging
import flwr as fl
import time
from functools import partial
from . import client_api

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

class FLClient(fl.client.NumPyClient):

    def __init__(self, model, validation_split, fl_task_id, client_mac, client_name, fl_round,next_gl_model, wandb_use, 
                 wandb_run=None, model_name=None, model_type=None, x_train=None, y_train=None, x_test=None, y_test=None, 
                 train_loader=None, val_loader=None, test_loader=None, criterion=None, optimizer=None, train_torch=None, test_torch=None):
        self.model_type = model_type
        self.model = model
        self.validation_split = validation_split
        self.fl_task_id = fl_task_id
        self.client_mac = client_mac
        self.client_name = client_name
        self.fl_round = fl_round
        self.next_gl_model = next_gl_model
        self.model_name = model_name
        self.wandb_use = wandb_use
        
        if self.wandb_use:
            self.wandb_run = wandb_run
        
        if self.model_type == "Tensorflow": 
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
        
        elif self.model_type == "Pytorch":
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.train_torch = train_torch
            self.test_torch = test_torch


    def set_parameters(self, parameters):
        import torch
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k] # Excluding parameters of BN layers
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=True)
        self.model.load_state_dict(state_dict, strict=False)
    
    def get_parameters(self):
        """Get parameters of the local model."""
        if self.model_type == "Tensorflow":
            raise Exception("Not implemented (server-side parameter initialization)")
        
        elif self.model_type == "Pytorch":
            # Excluding parameters of BN layers
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "bn" not in name]

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        print(f"config: {config}")
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

        if self.wandb_use:
            # add wandb config
            self.wandb_run.config.update({"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds}, allow_val_change=True)

        # start round time
        round_start_time = time.time()

        # model path for saving local model
        model_path = f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.next_gl_model}'

        # Initialize results
        results = {}
        
        # Training Tensorflow
        if self.model_type == "Tensorflow":
            # Update local model parameters
            self.model.set_weights(parameters)
            
            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=self.validation_split,
            )

            train_loss = history.history["loss"][len(history.history["loss"])-1]
            train_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
            results = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
                "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
            }

            # Return updated model parameters
            parameters_prime = self.model.get_weights()
            num_examples_train = len(self.x_train)

            
            # save local model
            self.model.save(model_path+'.h5')
            

        # Training Torch
        elif self.model_type == "Pytorch":
            # Update local model parameters
            self.set_parameters(parameters)
            
            trained_model = self.train_torch(self.model, self.train_loader, self.criterion, self.optimizer, 
                                            epochs)
            
            train_loss, train_accuracy, train_metrics = self.test_torch(trained_model, self.train_loader, self.criterion)
            val_loss, val_accuracy, val_metrics = self.test_torch(trained_model, self.val_loader, self.criterion)
            
            if train_metrics!=None:
                train_results = {"loss": train_loss,"accuracy": train_accuracy,**train_metrics}
                val_results = {"loss": val_loss,"accuracy": val_accuracy, **val_metrics}
            else:
                train_results = {"loss": train_loss,"accuracy": train_accuracy}
                val_results = {"loss": val_loss,"accuracy": val_accuracy}
                
            # Prefixing keys with 'train_' and 'val_'
            train_results_prefixed = {"train_" + key: value for key, value in train_results.items()}
            val_results_prefixed = {"val_" + key: value for key, value in val_results.items()}

            # Return updated model parameters
            parameters_prime = self.get_parameters()
            num_examples_train = len(self.train_loader)
            
            # Save model weights
            import torch
            torch.save(self.model.state_dict(), model_path+'.pth')
        else:
            raise ValueError("Unsupported model_type")


        # end round time
        round_end_time = time.time() - round_start_time

        if self.wandb_use:
            # wandb train log
            self.wandb_run.log({"train_time": round_end_time, "round": self.fl_round}, step=self.fl_round)  # train time

            # Log training results
            for key, value in train_results_prefixed.items():
                self.wandb_run.log({key: value, "round": self.fl_round}, step=self.fl_round)

            # Log validation results
            for key, value in val_results_prefixed.items():
                self.wandb_run.log({key: value, "round": self.fl_round}, step=self.fl_round)


        # Training: model performance by round
        results = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round, "next_gl_model_v": self.next_gl_model,
                    "train_time": round_end_time, **train_results_prefixed, **val_results_prefixed}

        json_result = json.dumps(results)
        logger.info(f'train_performance - {json_result}')

        # send train_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        return parameters_prime, num_examples_train, {**train_results_prefixed, **val_results_prefixed}


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Get config values
        batch_size: int = config["batch_size"]

        # Initialize test_loss, test_accuracy
        test_loss = 0.0
        test_accuracy = 0.0
        
        metrics=None
        
        if self.model_type == "Tensorflow":
            # Update local model with global parameters
            self.model.set_weights(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)

            num_examples_test = len(self.x_test)
            
        elif self.model_type == "Pytorch":            
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy, metrics = self.test_torch(self.model, self.test_loader, self.criterion)
            num_examples_test = len(self.test_loader)
        else:
            raise ValueError("Unsupported model_type")

        if self.wandb_use:
            # wandb log
            self.wandb_run.log({"test_loss": test_loss, "round": self.fl_round}, step=self.fl_round)  # loss
            self.wandb_run.log({"test_acc": test_accuracy, "round": self.fl_round}, step=self.fl_round)  # acc
            
            if metrics!=None:
                # Log other metrics dynamically
                for metric_name, metric_value in metrics.items():
                    self.wandb_run.log({metric_name: metric_value}, step=self.fl_round)

        # Test: model performance by round
        if metrics!=None:
            test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
                        "test_loss": test_loss, "test_acc": test_accuracy, **metrics, "next_gl_model_v": self.next_gl_model}
        else:
            test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
                        "test_loss": test_loss, "test_acc": test_accuracy, "next_gl_model_v": self.next_gl_model}
        json_result = json.dumps(test_result)
        logger.info(f'test - {json_result}')

        # send test_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_test_result(json_result)

        # increase next round
        self.fl_round += 1

        if metrics!=None:
            return test_loss, num_examples_test, {"accuracy": test_accuracy, **metrics}
        else:
            return test_loss, num_examples_test, {"accuracy": test_accuracy}


def flower_client_start(FL_server_IP, client):
    client_start = partial(fl.client.start_numpy_client, server_address=FL_server_IP, client=client)
    return client_start