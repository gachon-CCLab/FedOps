from collections import OrderedDict
import json, logging
from unittest import result
import flwr as fl
import time, os
from functools import partial
import numba
from numba import cuda
from . import client_api
from . import client_utils

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

    def __init__(self, model, validation_split, fl_task_id, client_mac, fl_round,next_gl_model, wandb_name, 
                 wandb_run=None, model_name=None, model_type=None, x_train=None, y_train=None, x_test=None, y_test=None, 
                 train_loader=None, val_loader=None, test_loader=None, criterion=None, optimizer=None, train_torch=None, test_torch=None):
        self.model_type = model_type
        self.model = model
        self.validation_split = validation_split
        self.fl_task_id = fl_task_id
        self.client_mac = client_mac
        self.fl_round = fl_round
        self.next_gl_model = next_gl_model
        self.wandb_name = wandb_name
        self.wandb_run = wandb_run
        self.model_name = model_name
        
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
        
        # Check if GPU is available
        numba.cuda.select_device(0)
        device = numba.cuda.get_current_device()

        if device != None:
            print("GPU is available.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            # CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("GPU not available, using CPU")


    def set_parameters(self, parameters):
        import torch
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=True)
        self.model.load_state_dict(state_dict)
    
    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

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
            
            # results = client_utils.torch_train(self.model, self.train_dataset, self.criterion, self.optimizer,
            #                                     self.validation_split, epochs, batch_size)
            trained_model = self.train_torch(self.model, self.train_loader, self.criterion, self.optimizer, 
                                            epochs)
            
            train_loss, train_accuracy = self.test_torch(trained_model, self.train_loader, self.criterion)
            val_loss, val_accuracy = self.test_torch(trained_model, self.val_loader, self.criterion)
            
            results = {"train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    }
            
            # Return updated model parameters
            parameters_prime = client_utils.get_model_params(self.model)
            num_examples_train = len(self.train_loader)
            
            # Save model weights
            import torch
            torch.save(self.model.state_dict(), model_path+'.pth')
        else:
            raise ValueError("Unsupported model_type")


        # end round time
        round_end_time = time.time() - round_start_time


        # wandb train log
        self.wandb_run.log({"train_loss": results["train_loss"], "round": self.fl_round}, step=self.fl_round) # train_loss
        self.wandb_run.log({"train_accuracy": results["train_accuracy"], "round": self.fl_round}, step=self.fl_round)  # train_accuracy
        self.wandb_run.log({"train_time": round_end_time, "round": self.fl_round}, step=self.fl_round)  # train time

        # wandb test log
        self.wandb_run.log({"val_loss": results["val_loss"], "round": self.fl_round}, step=self.fl_round) # test_loss
        self.wandb_run.log({"val_accuracy": results["val_accuracy"], "round": self.fl_round}, step=self.fl_round)  # test_accuracy


        # Training: model performance by round
        train_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "round": self.fl_round, "train_loss": results["train_loss"], "train_accuracy": results["train_accuracy"],
                        "test_loss": results["val_loss"], "test_accuracy": results["val_accuracy"], "train_time": round_end_time, "next_gl_model_v": self.next_gl_model, "wandb_name": self.wandb_name}
        json_result = json.dumps(train_result)
        logger.info(f'train_performance - {json_result}')

        # send train_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        return parameters_prime, num_examples_train, results


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Get config values
        steps: int = config["val_steps"]
        batch_size: int = config["batch_size"]

        # Initialize test_loss, test_accuracy
        test_loss = 0.0
        test_accuracy = 0.0
        
        if self.model_type == "Tensorflow":
            # Update local model with global parameters
            self.model.set_weights(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size, steps=steps)

            num_examples_test = len(self.x_test)
            
        elif self.model_type == "Pytorch":            
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy = self.test_torch(self.model, self.test_loader, self.criterion, steps)
            num_examples_test = len(self.test_loader)
        else:
            raise ValueError("Unsupported model_type")

        # wandb log
        self.wandb_run.log({"test_loss": test_loss, "round": self.fl_round}, step=self.fl_round)  # loss
        self.wandb_run.log({"test_acc": test_accuracy, "round": self.fl_round}, step=self.fl_round)  # acc

        # Test: model performance by round
        test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "round": self.fl_round,
                       "test_loss": test_loss, "test_acc": test_accuracy, "next_gl_model_v": self.next_gl_model, "wandb_name": self.wandb_name}
        json_result = json.dumps(test_result)
        logger.info(f'test - {json_result}')

        # send test_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_test_result(json_result)

        # increase next round
        self.fl_round += 1

        return test_loss, num_examples_test, {"accuracy": test_accuracy}


def flower_client_start(FL_server_IP, client):
    client_start = partial(fl.client.start_numpy_client, server_address=FL_server_IP, client=client)
    return client_start