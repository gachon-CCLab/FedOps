#client_fl.py
from collections import OrderedDict
import json, logging
import flwr as fl
import time
import os
from functools import partial
from . import client_api 
from . import client_utils

# set log format
handlers_list = [logging.StreamHandler()]

if "MONITORING" in os.environ:
    if os.environ["MONITORING"] == '1':
        handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
    else:
        pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)
import warnings
import torch

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

class FLClient(fl.client.NumPyClient):

    def __init__(self, model, validation_split, fl_task_id, client_mac, client_name, fl_round,gl_model, wandb_use, wandb_name,
                 wandb_run=None, model_name=None, model_type=None, x_train=None, y_train=None, x_test=None, y_test=None, 
                 train_loader=None, val_loader=None, test_loader=None, cfg=None, train_torch=None, test_torch=None,
                 finetune_llm=None, trainset=None, tokenizer=None, data_collator=None, formatting_prompts_func=None, num_rounds=None):
        
        self.cfg = cfg
        self.model_type = model_type
        self.model = model
        self.validation_split = validation_split
        self.fl_task_id = fl_task_id
        self.client_mac = client_mac
        self.client_name = client_name
        self.fl_round = fl_round
        self.gl_model = gl_model
        self.model_name = model_name
        self.wandb_use = wandb_use
        self.wandb_run = wandb_run
        self.wandb_name = wandb_name            
        
        if self.model_type == "Tensorflow": 
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
        
        elif self.model_type == "Pytorch":
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.train_torch = train_torch
            self.test_torch = test_torch

        elif self.model_type == "Huggingface":
            self.trainset = trainset
            self.tokenizer = tokenizer
            self.finetune_llm = finetune_llm
            self.data_collator = data_collator
            self.formatting_prompts_func = formatting_prompts_func


    def set_parameters(self, parameters):
        if self.model_type in ["Tensorflow"]:
            raise Exception("Not implemented")
        
        elif self.model_type in ["Pytorch"]:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k] # Excluding parameters of BN layers
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # self.model.load_state_dict(state_dict, strict=True)
            self.model.load_state_dict(state_dict, strict=False)

        elif self.model_type in ["Huggingface"]:
            client_utils.set_parameters_for_llm(self.model, parameters)
    
    def get_parameters(self):
        """Get parameters of the local model."""
        if self.model_type == "Tensorflow":
            raise Exception("Not implemented (server-side parameter initialization)")
        
        elif self.model_type == "Pytorch":
            # Excluding parameters of BN layers
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "bn" not in name]
        
        elif self.model_type == "Huggingface":
            return client_utils.get_parameters_for_llm(self.model)

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # _target_: server.strategy_cluster_optuna.ClusterOptunaFedAvg 설정시
        has_cluster_id = "cluster_id" in config   # ✨
        cluster_id = int(config["cluster_id"]) if has_cluster_id else None  # ✨
        # cluster_id = int(config.get("cluster_id", 0))  
        if has_cluster_id and (cluster_id is not None):  # ✨
            try:
                client_api.ClientServerAPI(self.fl_task_id).put_cluster_assign(self.client_mac, cluster_id)  # ✨
            except Exception as e:  # ✨
                logger.warning(f"[cluster-upsert] failed: {e}")  # ✨

        # try:                                                                                 
        #     client_api.ClientServerAPI(self.fl_task_id).put_cluster_assign(                  
        #         self.client_mac, int(cluster_id) if cluster_id is not None else None         
        #     )                                                                                
        # except Exception as e:                                                               
        #     logger.warning(f"[cluster-upsert] failed: {e}")                                  

        # print(f"config: {config}")
        
        # Get hyperparameters for this round
        batch_size: int = config.get("batch_size", self.cfg.batch_size)
        epochs: int = config.get("local_epochs", self.cfg.num_epochs)
        num_rounds: int = config.get("num_rounds", self.cfg.num_rounds)

        # HPO override (서버 전략에서 내려준 값) 하이퍼파라미터 튜닝을 하게되면 바뀌니까.
        hp_lr = config.get("hp_learning_rate", None) # ✨
        hp_bs = config.get("hp_batch_size", None) # ✨
        hp_ep = config.get("hp_local_epochs", None) # ✨
        if hp_bs is not None: # ✨
            batch_size = int(hp_bs) # ✨
        if hp_ep is not None: # ✨
            epochs = int(hp_ep) # ✨
 
        if self.wandb_use:
            # add wandb config
            wb_cfg = {"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds} 
            if hp_lr is not None: wb_cfg["hp_learning_rate"] = float(hp_lr) # ✨
            if hp_bs is not None: wb_cfg["hp_batch_size"] = int(hp_bs) # ✨ 
            if hp_ep is not None: wb_cfg["hp_local_epochs"] = int(hp_ep) # ✨
            self.wandb_run.config.update(wb_cfg, allow_val_change=True)

        # start round time
        round_start_time = time.time()

        # model path for saving local model
        model_path = f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.gl_model}'

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
            from torch.utils.data import DataLoader

            # Update local model parameters
            self.set_parameters(parameters)

            # clustering change
            train_loader = self.train_loader
            if hp_bs is not None:
                train_loader = DataLoader(self.train_loader.dataset, batch_size=int(hp_bs), shuffle=True)

            # lr override 전달
            hp = {}
            if hp_lr is not None:
                hp["learning_rate"] = float(hp_lr)
             # clustering change end
            trained_model = self.train_torch(self.model, train_loader, epochs, self.cfg, hp=hp)
            
            train_loss, train_accuracy, train_metrics = self.test_torch(trained_model, train_loader, self.cfg)
            val_loss, val_accuracy, val_metrics = self.test_torch(trained_model, self.val_loader, self.cfg)
            
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
           
            num_examples_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader)
            
            # Save model weights
            import torch
            torch.save(self.model.state_dict(), model_path+'.pth')

        elif self.model_type == "Huggingface":
            train_results_prefixed = {}
            val_results_prefixed = {}

            # Update local model parameters: LoRA Adapter params
            self.set_parameters(parameters)
            trained_model = self.finetune_llm(self.model, self.trainset, self.tokenizer, self.formatting_prompts_func, self.data_collator)
            parameters_prime = self.get_parameters()
            num_examples_train = len(self.trainset)

            train_loss = results["train_loss"] if "train_loss" in results else None
            results = {"train_loss": train_loss}

            model_save_path = model_path
            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)

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
        
        # ✨
        results = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round, "gl_model_v": self.gl_model,
                        **train_results_prefixed, **val_results_prefixed,"train_time": round_end_time, 'wandb_name': self.wandb_name,}
        if cluster_id is not None:
            results["cluster_id"] = int(cluster_id)

        json_result = json.dumps(results)
        logger.info(f'train_performance - {json_result}')

        # send train_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        return parameters_prime, num_examples_train, {**train_results_prefixed, **val_results_prefixed}


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        cluster_id = config.get("cluster_id", None)
        # cluster_id = int(config.get("cluster_id", 0))
        # Get config values
        batch_size: int = config.get("batch_size", self.cfg.batch_size)

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
            test_loss, test_accuracy, metrics = self.test_torch(self.model, self.test_loader, self.cfg)
            num_examples_test = len(self.test_loader.dataset) if hasattr(self.test_loader, "dataset") else len(self.test_loader)
        
        elif self.model_type == "Huggingface":
            test_loss = 0.0
            test_accuracy = 0.0
            num_examples_test = 1
        
        else:
            raise ValueError("Unsupported model_type")

        if self.wandb_use:
            # wandb log
            self.wandb_run.log({"test_loss": test_loss, "round": self.fl_round}, step=self.fl_round)  # loss
            self.wandb_run.log({"test_accuracy": test_accuracy, "round": self.fl_round}, step=self.fl_round)  # acc
            
            if metrics!=None:
                for metric_name, metric_value in metrics.items():
                    self.wandb_run.log({metric_name: metric_value}, step=self.fl_round)

        test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
                         "test_loss": test_loss, "test_accuracy": test_accuracy, "gl_model_v": self.gl_model, 'wandb_name': self.wandb_name,}
        if cluster_id is not None:
            test_result["cluster_id"] = int(cluster_id)
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