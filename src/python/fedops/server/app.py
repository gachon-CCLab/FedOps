import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
from . import server_api
from . import server_utils

# TF warning log filtering
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLServer():
    def __init__(self, config, model, model_name, x_val, y_val):
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID
        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.dataset = config['data']['name']
        self.fraction_fit = float(config['aggregation']['fedAvg']['fraction_fit'])
        self.fraction_evaluate = float(config['aggregation']['fedAvg']['fraction_evaluate'])
        self.min_fit_clients = int(config['aggregation']['fedAvg']['min_fit_clients'])
        self.min_evaluate_clients = int(config['aggregation']['fedAvg']['min_evaluate_clients'])
        self.min_available_clients = int(config['aggregation']['fedAvg']['min_available_clients'])

        self.batch_size = int(config['fl_server']['batch_size'])
        self.local_epochs = int(config['fl_server']['local_epochs'])
        self.num_rounds = int(config['fl_server']['num_rounds'])
        self.val_steps = int(config['fl_server']['val_steps'])

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None
        self.x_val = x_val
        self.y_val = y_val



    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'latest_gl_model_v: {self.server.latest_gl_model_v}')

        if not model:

            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')

            self.fl_server_start(init_model, model_name)
            return model_name


        else:
            logging.info('load latest global model')
            print(f'latest_gl_model_name: {gl_model_name}')

            self.fl_server_start(model, gl_model_name)
            return gl_model_name


    def fl_server_start(self, model, model_name):
        # Load and compile model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
            evaluate_fn=self.get_eval_fn(model, model_name),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights())
        )

        # Start Flower server (SSL-enabled) for four rounds of federated learning
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )


    def get_eval_fn(self, model, model_name):
        """Return an evaluation function for server-side evaluation."""
        # Load data and model here to avoid the overhead of doing it in `evaluate` itself

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
            loss, accuracy = model.evaluate(self.x_val, self.y_val)

            model.set_weights(parameters)  # Update model with the latest parameters

            if self.server.round >= 1:
                # fit aggregation end time
                self.server.end_by_round = time.time() - self.server.start_by_round
                # gl model performance by round
                server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_acc": accuracy,
                                      "run_time_by_round": self.server.end_by_round, "next_gl_model_v":self.server.next_gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')

                # send gl model evaluation to performance pod
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)


            # model save
            model.save(f'/app/{model_name}_gl_model_V{self.server.next_gl_model_v}.h5')

            return loss, {"accuracy": accuracy}

        return evaluate


    def fit_config(self, rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }

        # increase round
        self.server.round += 1

        # if server.round > 2:
        # fit aggregation start time
        self.server.start_by_round = time.time()
        logging.info('server start by round')

        return fl_config


    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """

        return {"val_steps": self.val_steps}

    def start(self):

        today = datetime.datetime.today()
        today_time = today.strftime('%Y-%m-%d %H-%M-%S')

        # global model init
        # model = None
        # server.latest_gl_model_v = 0

        # Loaded latest global model or no global model
        self.next_model, self.next_model_name, self.server.latest_gl_model_v = server_utils.model_download(self.task_id)
        # logging.info('Loaded latest global model or no global model')

        # New Global Model Version
        self.server.next_gl_model_v = self.server.latest_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            'S3_bucket': 'fl-gl-model',  # bucket name
            'Latest_GL_Model': 'gl_model_%s_V.h5' % self.server.latest_gl_model_v,  # Model Weight File Name
            'Play_datetime': today_time,
            'FLSeReady': True,  # server ready status
            'GL_Model_V': self.server.latest_gl_model_v  # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "next_gl_model_v": self.server.next_gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)

            # upload global model
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.next_gl_model_v}.h5"
            server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)

            logging.info(f'upload {global_model_file_name} model in s3')

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