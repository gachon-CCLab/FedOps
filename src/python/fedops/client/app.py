import logging, json
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
    def __init__(self, config, fl_task):
        self.app = FastAPI()
        
        self.x_train = fl_task[0]
        self.x_test = fl_task[1]
        self.y_train = fl_task[2]
        self.y_test = fl_task[3]
        self.y_label_counter = fl_task[4]
        self.model = fl_task[5]
        self.model_name = fl_task[6]
        self.task_id = config['task']['name']
        self.dataset = config['data']['name']
        self.label_count = config['data']['label_count']
        self.validation_split = config['data']['validation_split']
        self.wandb_key = config['wandb']['api_key']
        self.wandb_account = config['wandb']['account']
        
        self.status = client_utils.FLClientStatus()

    async def fl_client_start(self):
        logging.info('FL learning ready')

        logging.info(f'fl_task_id: {self.task_id}')
        logging.info(f'dataset: {self.dataset}')
        logging.info(f'label_count: {self.label_count}')
        logging.info(f'validation_split: {self.validation_split}')
        logging.info(f'wandb_key: {self.wandb_key}')
        logging.info(f'wandb_account: {self.wandb_account}')

        """
        Before running this code,
        set wandb api and account in the config.yaml
        """
        # Set the name in the wandb project
        wandb_name = f"client-v{self.status.FL_next_gl_model}({datetime.now()})"
        # Login and init wandb project
        wandb_run = client_wandb.start_wandb(self.wandb_key, self.task_id, wandb_name)

        try:
            # Update wandb config
            wandb_config = {"learning_rate": self.model.optimizer.learning_rate.numpy(),
                            "optimizer": self.model.optimizer,
                            "dataset": self.dataset, "model_architecture": self.model_name}
            wandb_run.config.update(wandb_config, allow_val_change=True)

            # Check data fl client data status in the wandb
            label_values = [[i, self.y_label_counter[i]] for i in range(self.label_count)]
            client_wandb.data_status_wandb(wandb_run, label_values)

            loop = asyncio.get_event_loop()
            client = client_fl.FLClient(self.model, self.x_train, self.y_train, self.x_test,
                                        self.y_test,
                                        self.validation_split, self.task_id, self.status.FL_client_mac,
                                        fl_round=1, next_gl_model=self.status.FL_next_gl_model, wandb_name=wandb_name,
                                        wandb_run=wandb_run, model_name=self.model_name)

            # client_start object
            client_start = client_fl.flower_client_start(self.status.FL_server_IP, client)

            # FL client start time
            fl_start_time = time.time()

            # Run asynchronously FL client
            await loop.run_in_executor(None, client_start)

            logging.info('fl learning finished')

            # FL client end time
            fl_end_time = time.time() - fl_start_time

            # Wandb log(Client round end time)
            wandb_run.log({"operation_time": fl_end_time, "next_gl_model_v": self.status.FL_next_gl_model},
                          step=self.status.FL_next_gl_model)

            client_all_time_result = {"fl_task_id": self.task_id, "client_mac": self.status.FL_client_mac,
                                      "operation_time": fl_end_time,
                                      "next_gl_model_v": self.status.FL_next_gl_model, "wandb_name": wandb_name}
            json_result = json.dumps(client_all_time_result)
            logging.info(f'client_operation_time - {json_result}')

            # Send client_time_result to client_performance pod
            client_api.ClientServerAPI(self.task_id).put_client_time_result(json_result)

            # Get client system result from wandb and send it to client_performance pod
            client_wandb.client_system_wandb(self.task_id, self.status.FL_client_mac, self.status.FL_next_gl_model,
                                             wandb_name, self.wandb_account)

            # Delete client object
            del client

            # Complete Client training
            self.status.FL_client_start = await client_utils.notify_fin()
            logging.info('FL Client Learning Finish')

        except Exception as e:
            logging.info('[E][PC0002] learning', e)
            self.status.FL_client_fail = True
            self.status.FL_client_start = await client_utils.notify_fail()
            raise e

    def start(self):
        # Configure routes, endpoints, and other FastAPI-related logic here
        # Example:
        @self.app.get('/online')
        async def get_info():
            return self.status

        # asynchronously start client
        @self.app.post("/start")
        async def client_start_trigger(background_tasks: BackgroundTasks, manager_data: client_utils.ManagerData):

            logging.info(f'fl_task_id: {self.task_id}')
            logging.info(f'dataset: {self.dataset}')
            logging.info(f'label_count: {self.label_count}')
            logging.info(f'validation_split: {self.validation_split}')
            logging.info(f'wandb_key: {self.wandb_key}')
            logging.info(f'wandb_account: {self.wandb_account}')

            # client_manager address
            client_res = client_api.ClientMangerAPI().get_info()

            # latest global model version
            latest_gl_model_v = client_res.json()['GL_Model_V']

            # next global model version
            self.status.FL_next_gl_model = latest_gl_model_v + 1

            logging.info('bulid model')

            logging.info('FL start')
            self.status.FL_client_start = True

            self.status.FL_server_IP = manager_data.server_ip
            self.status.FL_task_id = self.task_id
            self.status.FL_client_mac = client_utils.get_mac_address()

            # get the FL server IP
            self.status.FL_server_IP = client_api.ClientServerAPI(self.task_id).get_port()

            # start FL Client
            background_tasks.add_task(self.fl_client_start)

            return self.status
        
        try:
            # create client api => to connect client manager
            uvicorn.run(self.app, host='0.0.0.0', port=8002)

        except Exception as e:
            # Handle any exceptions that occur during the execution
            logging.error(f'An error occurred during execution: {e}')

        finally:
            # FL client out
            client_api.ClientMangerAPI().get_client_out()
            logging.info(f'{self.status.FL_client_mac}-client close')
            

