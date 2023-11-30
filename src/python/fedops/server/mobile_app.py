import logging
from typing import Dict, Optional, Tuple, cast
import flwr as fl
import datetime
import os
import json
import time
from . import server_api
from . import server_utils
from hydra.utils import instantiate

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLMobileServer():
    def __init__(self, cfg):
        
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID
        self.client_device = cfg.client_device

        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)


    def init_gl_model_registration(self) -> None:
        logging.info(f'FL Mobile Server Start')

        self.fl_server_start()

    def fit_config(self, server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            # "num_rounds": cfg.num_rounds,
        }
        return config

    def fl_server_start(self):
        # Create FL Server Strategy
        strategy = instantiate(
            self.strategy,
            on_fit_config_fn=self.fit_config,
        )
        
        # Start Flower server (SSL-enabled) for four rounds of federated learning
        hist = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )


    def start(self):

        today = datetime.datetime.today()
        today_time = today.strftime('%Y-%m-%d %H-%M-%S')

        # self.next_model, self.next_model_name, self.server.latest_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)

        # New Global Model Version
        # self.server.next_gl_model_v = self.server.latest_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            'S3_bucket': None,  # bucket name
            'Latest_GL_Model': None,  # Model Weight File Name
            'Play_datetime': today_time,
            'FLSeReady': True,  # server ready status
            'GL_Model_V': None # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            self.init_gl_model_registration()

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "next_gl_model_v": None}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # # # upload global model
            # # if self.model_type == "Tensorflow":
            # #     global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.next_gl_model_v}.h5"
            # # elif self.model_type =="Pytorch":
            # #     global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.next_gl_model_v}.pth"
            # # server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)

            # logging.info(f'upload {global_model_file_name} model in s3')

        # server_status error
        except Exception as e:
            logging.error('error: ', e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))

        finally:
            logging.info('server close')

            # Modifying the model version in server manager
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')
            # res = server_api.ServerAPI(task_id).put_fl_round_fin()
            # if res.status_code == 200:
            #     logging.info('global model version upgrade')
            # logging.info('global model version: ', res.json()['Server_Status']['GL_Model_V'])