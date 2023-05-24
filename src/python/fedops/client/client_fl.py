import json, logging

import flwr as fl
import time, os
from functools import partial

import tensorflow as tf
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

    def __init__(self, model, x_train, y_train, x_test, y_test, validation_split,fl_task_id, client_mac, fl_round,
                 next_gl_model, wandb_name, wandb_run, model_name):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.validation_split = validation_split
        self.fl_task_id = fl_task_id
        self.client_mac = client_mac
        self.fl_round = fl_round
        self.next_gl_model = next_gl_model
        self.wandb_name = wandb_name
        self.wandb_run = wandb_run
        self.model_name = model_name

        # Check if GPU is available
        if tf.test.is_gpu_available():
            # Set TensorFlow to use only the first GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            device_name = "/gpu:0"
            print("GPU available")
        else:
            # CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("GPU not available, using CPU")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

        # add wandb config
        self.wandb_run.config.update({"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds}, allow_val_change=True)

        # start round time
        round_start_time = time.time()

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=self.validation_split,
        )

        # end round time
        round_end_time = time.time() - round_start_time

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)

        FL_loss = history.history["loss"][len(history.history["loss"])-1]
        FL_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
        results = {
            "loss": FL_loss,
            "accuracy": FL_accuracy,
            "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
            "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
        }

        # wandb log
        self.wandb_run.log({"train_loss": FL_loss, "round": self.fl_round}, step=self.fl_round) # loss
        self.wandb_run.log({"train_acc": FL_accuracy, "round": self.fl_round}, step=self.fl_round)  # acc
        self.wandb_run.log({"train_time": round_end_time, "round": self.fl_round}, step=self.fl_round)  # train time

        # Training: model performance by round
        train_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "round": self.fl_round, "train_loss": FL_loss, "train_acc": FL_accuracy,
                        "train_time": round_end_time, "next_gl_model_v": self.next_gl_model, "wandb_name": self.wandb_name}
        json_result = json.dumps(train_result)
        logger.info(f'train_performance - {json_result}')

        # send train_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        # save local model
        self.model.save(f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.next_gl_model}.h5')

        return parameters_prime, num_examples_train, results


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=1024, steps=steps)
        num_examples_test = len(self.x_test)

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