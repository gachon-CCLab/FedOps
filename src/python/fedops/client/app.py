import logging, json
import socket
import time
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from datetime import datetime

from . import client_utils
from . import client_fl
from . import client_wandb
from . import client_api


class FLClientTask():
    def __init__(self, cfg, fl_task=None):
        self.app = FastAPI()
        self.status = client_utils.FLClientStatus()
        self.cfg = cfg
        self.client_port = 8003
        self.task_id = cfg.task_id
        self.dataset_name = cfg.dataset.name
        self.output_size = cfg.model.output_size
        self.validation_split = cfg.dataset.validation_split
        self.wandb_use = cfg.wandb.use
        self.model_type = cfg.model_type
        self.model = fl_task["model"]
        self.model_name = fl_task["model_name"]
        self.y_label_counter = fl_task["y_label_counter"]
        
        logging.info(f'init model_type: {self.model_type}')
        
        if self.wandb_use:
            self.wandb_key = cfg.wandb.key
            self.wandb_account = cfg.wandb.account
            self.wandb_project = cfg.wandb.project

        if self.model_type=="Tensorflow":
            self.x_train = fl_task["x_train"]
            self.x_test = fl_task["x_test"]
            self.y_train = fl_task["y_train"]
            self.y_test = fl_task["y_test"]

        elif self.model_type == "Pytorch":
            self.train_loader = fl_task["train_loader"]
            self.val_loader = fl_task["val_loader"]
            self.test_loader = fl_task["test_loader"]
            self.criterion = fl_task["criterion"]
            self.optimizer = fl_task["optimizer"]
            self.train_torch = fl_task["train_torch"]
            self.test_torch = fl_task["test_torch"]
                    

    async def fl_client_start(self):
        logging.info('FL learning ready')

        logging.info(f'fl_task_id: {self.task_id}')
        logging.info(f'dataset: {self.dataset_name}')
        logging.info(f'output_size: {self.output_size}')
        logging.info(f'validation_split: {self.validation_split}')
        logging.info(f'model_type: {self.model_type}')

        """
        Before running this code,
        set wandb api and account in the config.yaml
        """
        if self.wandb_use:
            logging.info(f'wandb_key: {self.wandb_key}')
            logging.info(f'wandb_account: {self.wandb_account}')
            # Set the name in the wandb project
            wandb_name = f"client-v{self.status.next_gl_model}({datetime.now()})"
            # Login and init wandb project
            wandb_run = client_wandb.start_wandb(self.wandb_key, self.wandb_project, wandb_name)
        else:
            wandb_run=None
        
        # Initialize wandb_config, client object
        wandb_config = {}
        # client = None
        
        try:
            loop = asyncio.get_event_loop()
            if self.model_type == "Tensorflow":
                # Update wandb config
                wandb_config = {"learning_rate": self.model.optimizer.learning_rate.numpy(),
                                "optimizer": self.model.optimizer,
                                "loss_function": self.model.loss,
                                "dataset": self.dataset_name, "model_architecture": self.model_name}

                client = client_fl.FLClient(model=self.model, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test,
                                            y_test=self.y_test,
                                            validation_split=self.validation_split, fl_task_id=self.task_id, client_mac=self.status.client_mac, 
                                            client_name=self.status.client_name,
                                            fl_round=1, next_gl_model=self.status.next_gl_model, wandb_use=self.wandb_use,
                                            wandb_run=wandb_run, model_name=self.model_name, model_type=self.model_type)

            elif self.model_type == "Pytorch":
                wandb_config = {"learning_rate": self.optimizer.param_groups[0]['lr'],
                                "optimizer": self.optimizer.__class__.__name__,
                                "loss_function": self.criterion,
                                "dataset": self.dataset_name, "model_architecture": self.model_name}

                client = client_fl.FLClient(model=self.model, validation_split=self.validation_split, 
                                            fl_task_id=self.task_id, client_mac=self.status.client_mac, client_name=self.status.client_name,
                                            fl_round=1, next_gl_model=self.status.next_gl_model, wandb_use=self.wandb_use,
                                            wandb_run=wandb_run, model_name=self.model_name, model_type=self.model_type, 
                                            train_loader=self.train_loader, val_loader=self.val_loader, test_loader=self.test_loader, 
                                            criterion=self.criterion, optimizer=self.optimizer, 
                                            train_torch=self.train_torch, test_torch=self.test_torch)


            
            # Check data fl client data status in the wandb
            label_values = [[i, self.y_label_counter[i]] for i in range(self.output_size)]
            logging.info(f'label_values: {label_values}')

            # client_start object
            client_start = client_fl.flower_client_start(self.status.server_IP, client)

            # FL client start time
            fl_start_time = time.time()

            # Run asynchronously FL client
            await loop.run_in_executor(None, client_start)

            logging.info('fl learning finished')

            # FL client end time
            fl_end_time = time.time() - fl_start_time
            
            
            if self.wandb_use:
                wandb_run.config.update(wandb_config, allow_val_change=True)
                client_wandb.data_status_wandb(wandb_run, label_values)
                # Wandb log(Client round end time)
                wandb_run.log({"operation_time": fl_end_time, "next_gl_model_v": self.status.next_gl_model},step=self.status.next_gl_model)
                # close wandb
                wandb_run.finish()
                
                # Get client system result from wandb and send it to client_performance pod
                client_wandb.client_system_wandb(self.task_id, self.status.client_mac, self.status.client_name, 
                                                 self.status.next_gl_model, wandb_name, self.wandb_account)

            client_all_time_result = {"fl_task_id": self.task_id, "client_mac": self.status.client_mac, "client_name": self.status.client_name,
                                      "operation_time": fl_end_time,
                                      "next_gl_model_v": self.status.next_gl_model}
            json_result = json.dumps(client_all_time_result)
            logging.info(f'client_operation_time - {json_result}')

            # Send client_time_result to client_performance pod
            client_api.ClientServerAPI(self.task_id).put_client_time_result(json_result)

            # Delete client object
            del client

            # Complete Client training
            self.status.client_start = await client_utils.notify_fin()
            logging.info('FL Client Learning Finish')

        except Exception as e:
            logging.info('[E][PC0002] learning', e)
            self.status.client_fail = True
            self.status.client_start = await client_utils.notify_fail()
            raise e

    def start(self):
        # Configure routes, endpoints, and other FastAPI-related logic here
        # Example:
        @self.app.get('/online')
        async def get_info():
            return self.status

        # asynchronously start client
        @self.app.post("/start")
        async def client_start_trigger(background_tasks: BackgroundTasks):

            # client_manager address
            client_res = client_api.ClientMangerAPI().get_info()

            # # # latest global model version
            latest_gl_model_v = client_res.json()['GL_Model_V']

            # # next global model version
            self.status.next_gl_model = latest_gl_model_v + 1
            # self.status.next_gl_model = 1

            logging.info('bulid model')

            logging.info('FL start')
            self.status.client_start = True

            self.status.task_id = self.task_id
            self.status.client_mac = client_utils.get_mac_address()
            self.status.client_name = socket.gethostname()

            # get the FL server IP
            self.status.server_IP = client_api.ClientServerAPI(self.task_id).get_port()
            # self.status.server_IP = "0.0.0.0:8080"

            # start FL Client
            background_tasks.add_task(self.fl_client_start)

            return self.status

        try:
            # create client api => to connect client manager
            uvicorn.run(self.app, host='0.0.0.0', port=self.client_port)

        except Exception as e:
            # Handle any exceptions that occur during the execution
            logging.error(f'An error occurred during execution: {e}')

        finally:
            # FL client out
            client_api.ClientMangerAPI().get_client_out()
            logging.info(f'{self.status.client_name};{self.status.client_mac}-client close')


