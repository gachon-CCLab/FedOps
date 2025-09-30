# server/app.py

import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
import numpy as np
import shutil
from . import server_api
from . import server_utils
from collections import OrderedDict
from hydra.utils import instantiate

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, NDArrays
from ..utils.fedco.best_keeper import BestKeeper

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLServer():
    def __init__(self, cfg, model, model_name, model_type, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID

        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.model_type = model_type
        self.cfg = cfg
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None
        
        if self.model_type=="Tensorflow":
            self.x_val = x_val
            self.y_val = y_val  
               
        elif self.model_type == "Pytorch":
            self.gl_val_loader = gl_val_loader
            self.test_torch = test_torch

        elif self.model_type == "Huggingface":
            pass

        # ====== 클러스터 전략 여부/메트릭 키 결정 (비클러스터는 영향 없음) ======
        try:
            strat_target = str(self.strategy._target_)
        except Exception:
            strat_target = ""
        self.is_cluster = "server.strategy_cluster_optuna.ClusterOptunaFedAvg" in strat_target

        metric_key = "accuracy"
        if self.is_cluster:
            # yaml: server.strategy.objective -> maximize_f1 | maximize_acc | minimize_loss
            try:
                objective = str(getattr(self.strategy, "objective", "")).lower()
            except Exception:
                objective = ""
            if "maximize_f1" in objective:
                metric_key = "val_f1_score"
            elif "minimize_loss" in objective:
                metric_key = "val_loss"
            else:
                metric_key = "accuracy"

        # 클러스터일 때만 BestKeeper 활성화
        self.best_keeper = BestKeeper(save_dir="./gl_best", metric_key=metric_key) if self.is_cluster else None
        # ===============================================================


    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'last_gl_model_v: {self.server.last_gl_model_v}')

        if not model:

            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')

            self.fl_server_start(init_model, model_name)
            return model_name

        else:
            logging.info('load last global model')
            print(f'last_gl_model_name: {gl_model_name}')

            self.fl_server_start(model, gl_model_name)
            return gl_model_name


    def fl_server_start(self, model, model_name):
        # Load and compile model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation

        model_parameters = None # Init model_parametes variable
        
        if self.model_type == "Tensorflow":
            model_parameters = model.get_weights()
        elif self.model_type == "Pytorch":
            model_parameters = [val.cpu().detach().numpy() for _, val in model.state_dict().items()]
        elif self.model_type == "Huggingface":
            json_path = "./parameter_shapes.json"
            model_parameters = server_utils.load_initial_parameters_from_shape(json_path)

        strategy = instantiate(
            self.strategy,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            evaluate_fn=self.get_eval_fn(model, model_name),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        
        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )

        # ===== 학습 종료 후: (클러스터 전략일 때만) 최고 전역모델로 최종 파일 덮어쓰기 =====
        if self.is_cluster and self.best_keeper is not None:
            try:
                best_params = self.best_keeper.load_params()
                if best_params is not None:
                    gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'

                    if self.model_type == "Pytorch":
                        import torch
                        best_nds = parameters_to_ndarrays(best_params)
                        keys = [k for k in model.state_dict().keys() if "bn" not in k]
                        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, best_nds)})
                        model.load_state_dict(state_dict, strict=True)
                        torch.save(model.state_dict(), gl_model_path + '.pth')
                        logger.info("[BEST] Saved best global model to %s.pth", gl_model_path)

                        # (선택) 중앙 검증 로그
                        try:
                            loss_b, acc_b, met_b = self.test_torch(model, self.gl_val_loader, self.cfg)
                            logger.info(f"[FINAL-BEST] loss={loss_b:.4f}, acc={acc_b:.6f}, extra={met_b}")
                        except Exception:
                            pass

                    elif self.model_type == "Tensorflow":
                        best_nds = parameters_to_ndarrays(best_params)
                        model.set_weights(best_nds)
                        model.save(gl_model_path + '.h5')
                        logger.info("[BEST] Saved best global model to %s.h5", gl_model_path)

                        # (선택) 중앙 검증 로그
                        try:
                            loss_b, acc_b = model.evaluate(self.x_val, self.y_val, verbose=0)
                            logger.info(f"[FINAL-BEST] loss={loss_b:.4f}, acc={acc_b:.6f}")
                        except Exception:
                            pass

                    elif self.model_type == "Huggingface":
                        logger.info("[BEST] (HF) finalization skipped")
            except Exception as e:
                logger.error(f"[BEST] finalization error: {e}")


    def get_eval_fn(self, model, model_name):
        """Return an evaluation function for server-side evaluation."""
        # Load data and model here to avoid the overhead of doing it in `evaluate` itself

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters_ndarrays: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
                        
            # model path for saving global model snapshot by round
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            
            metrics = None
            
            if self.model_type == "Tensorflow":
                # 먼저 최신 파라미터 로드 후 평가
                model.set_weights(parameters_ndarrays)
                loss, accuracy = model.evaluate(self.x_val, self.y_val, verbose=0)
                model.save(gl_model_path + '.h5')
            
            elif self.model_type == "Pytorch":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                torch.save(model.state_dict(), gl_model_path + '.pth')

            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0
                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)

            # === 라운드별 로그/리포팅 (원래 로직 유지) ===
            if self.server.round >= 1:
                self.server.end_by_round = time.time() - self.server.start_by_round
                if metrics!=None:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round, **metrics,"gl_model_v":self.server.gl_model_v}
                else:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round,"gl_model_v":self.server.gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)
            
            # === (클러스터 전략일 때만) BestKeeper 갱신 ===
            if self.is_cluster and self.best_keeper is not None:
                merged_metrics = {"accuracy": accuracy}
                if metrics is not None:
                    merged_metrics.update(metrics)
                try:
                    self.best_keeper.update(
                        server_round=server_round,
                        parameters=ndarrays_to_parameters(parameters_ndarrays),
                        metrics=merged_metrics,
                    )
                except Exception as e:
                    logger.warning(f"[BEST] update skipped: {e}")

            if metrics!=None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
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

        # fit aggregation start time
        self.server.start_by_round = time.time()
        logging.info('server start by round')

        return fl_config


    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        """
        return {"batch_size": self.batch_size}


    def start(self):

        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        # Loaded last global model or no global model in s3
        self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)
        
        # Loaded last global model or no global model in local
        # self.next_model, self.next_model_name, self.server.latest_gl_model_v = server_utils.model_download_local(self.model_type, self.init_model)

        # New Global Model Version
        self.server.gl_model_v = self.server.last_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            "S3_bucket": "fl-gl-model",  # bucket name
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,  # Model Weight File Name
            "FLServer_start": today_time,
            "FLSeReady": True,  # server ready status
            "GL_Model_V": self.server.gl_model_v # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "gl_model_v": self.server.gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # upload global model (최종 파일은 비클러스터는 원래 파일, 클러스터는 BEST로 덮어쓴 파일)
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type =="Pytorch":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Huggingface":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
                npz_file_path = f"{global_model_file_name}.npz"
                model_dir = f"{global_model_file_name}"
                real_npz_path = os.path.join(model_dir, "adapter_parameters.npz")
                shutil.copy(real_npz_path, npz_file_path)
                server_utils.upload_model_to_bucket(self.task_id, npz_file_path)

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
