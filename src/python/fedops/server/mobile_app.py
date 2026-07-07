import logging
from typing import Dict, Optional, Tuple, cast
import flwr as fl
import datetime
import os
import json
import time
import threading
import signal
import re
from pathlib import Path
from . import server_api
from . import server_utils
from hydra.utils import instantiate
from flwr.common import parameters_to_ndarrays

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLMobileServer():
    def __init__(self, cfg):
        
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID
        self.client_device = cfg.get("client_device", "android")
        self.sba_fl = cfg.get("sba_fl", {})

        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)
        self.models_dir = Path("/app/data/models")
        self.logs_dir = Path("/app/data/logs")
        self.sba_round_metrics = []
        self.latest_global_model_path = None
        safe_task_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(self.task_id or "task")
        )
        run_stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.sba_run_id = f"{safe_task_id}_{run_stamp}"
        self.sba_run_started_at = None
        self.sba_run_finished_at = None
        self.sba_global_model_version_number = None
        self.sba_auto_shutdown_scheduled = False


    def init_gl_model_registration(self) -> None:
        logging.info(f'FL Mobile Server Start')
        logging.info(f'client_device: {self.client_device}')
        if self.sba_fl:
            logging.info(f'sba_fl config: {self.sba_fl}')

        self.fl_server_start()

    def fit_config(self, server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }
        return config

    def fl_server_start(self):
        # Create FL Server Strategy
        strategy = instantiate(
            self.strategy,
            on_fit_config_fn=self.fit_config,
        )
        if self.sba_fl:
            self._attach_sba_fl_artifact_hooks(strategy)
        
        # Start Flower server (SSL-enabled) for four rounds of federated learning
        hist = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
        if self.sba_fl:
            self._save_sba_fl_history(hist)

    def _attach_sba_fl_artifact_hooks(self, strategy):
        original_aggregate_fit = strategy.aggregate_fit
        original_aggregate_evaluate = strategy.aggregate_evaluate

        def aggregate_fit_with_artifact(server_round, results, failures):
            aggregated = original_aggregate_fit(server_round, results, failures)
            parameters, metrics = aggregated
            metric = self._ensure_sba_round_metric(server_round)
            metric["fit_results"] = len(results)
            metric["fit_failures"] = len(failures)
            metric["fit_metrics"] = metrics or {}
            if parameters is not None:
                model_path = self._save_sba_global_model(server_round, parameters)
                metric["model_path"] = str(model_path)
                self.latest_global_model_path = str(model_path)
            self._write_sba_history_file()
            return aggregated

        def aggregate_evaluate_with_artifact(server_round, results, failures):
            aggregated = original_aggregate_evaluate(server_round, results, failures)
            loss, metrics = aggregated
            metric = self._ensure_sba_round_metric(server_round)
            metric["evaluate_results"] = len(results)
            metric["evaluate_failures"] = len(failures)
            metric["distributed_loss"] = loss
            metric["evaluate_metrics"] = metrics or {}
            if int(server_round) >= self.num_rounds:
                self.sba_run_finished_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                self._write_sba_history_file(finished=True)
                self._schedule_sba_auto_shutdown()
            else:
                self._write_sba_history_file()
            return aggregated

        strategy.aggregate_fit = aggregate_fit_with_artifact
        strategy.aggregate_evaluate = aggregate_evaluate_with_artifact

    def _ensure_sba_round_metric(self, server_round):
        for metric in self.sba_round_metrics:
            if metric["round"] == server_round:
                return metric
        metric = {
            "round": int(server_round),
            "fit_results": None,
            "fit_failures": None,
            "evaluate_results": None,
            "evaluate_failures": None,
            "distributed_loss": None,
            "model_path": None,
        }
        self.sba_round_metrics.append(metric)
        self.sba_round_metrics.sort(key=lambda item: item["round"])
        return metric

    def _sba_weight_layout(self):
        return [
            ("lstm.weight_ih_l0", [128, 3]),
            ("lstm.weight_hh_l0", [128, 32]),
            ("lstm.bias_ih_l0", [128]),
            ("lstm.bias_hh_l0", [128]),
            ("fc1.weight", [32, 32]),
            ("fc1.bias", [32]),
            ("fc2.weight", [16, 32]),
            ("fc2.bias", [16]),
            ("fc3.weight", [1, 16]),
            ("fc3.bias", [1]),
        ]

    def _next_sba_global_model_version_number(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(r"global_model_latest_v(\d+)\.json$")
        max_version = 0
        for path in self.models_dir.glob("global_model_latest_v*.json"):
            match = pattern.match(path.name)
            if match:
                max_version = max(max_version, int(match.group(1)))
        return max_version + 1

    def _sba_versioned_latest_path(self):
        if self.sba_global_model_version_number is None:
            self.sba_global_model_version_number = self._next_sba_global_model_version_number()
        return self.models_dir / f"global_model_latest_v{self.sba_global_model_version_number}.json"

    def _schedule_sba_auto_shutdown(self, delay_seconds=2.0):
        if self.sba_auto_shutdown_scheduled:
            return
        self.sba_auto_shutdown_scheduled = True

        def request_shutdown():
            time.sleep(delay_seconds)
            logging.info("SBA_FL_AUTO_SHUTDOWN_REQUESTED after final round evaluation")
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=request_shutdown, daemon=True).start()

    def _save_sba_global_model(self, server_round, parameters):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        arrays = parameters_to_ndarrays(parameters)
        layout = self._sba_weight_layout()
        tensors = []
        for idx, array in enumerate(arrays):
            name, expected_shape = layout[idx] if idx < len(layout) else (f"tensor_{idx}", list(array.shape))
            tensors.append({
                "name": name,
                "shape": list(array.shape) if list(array.shape) else expected_shape,
                "values": array.astype("float32").reshape(-1).tolist(),
            })

        saved_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        is_final_round = int(server_round) >= self.num_rounds
        version_number = None
        version_label = None
        versioned_latest_path = None
        if is_final_round:
            versioned_latest_path = self._sba_versioned_latest_path()
            version_number = self.sba_global_model_version_number
            version_label = f"v{version_number}"

        root = {
            "modelType": "weight_lstm_parameters",
            "source": "fedops_sba_fl_global",
            "taskId": self.task_id,
            "runId": self.sba_run_id,
            "globalModelVersion": version_label,
            "globalModelVersionNumber": version_number,
            "round": int(server_round),
            "sequenceLength": 7,
            "featureCount": 3,
            "tensorCount": len(tensors),
            "tensors": tensors,
            "createdAt": saved_at,
            "savedAtIso": saved_at,
        }

        snapshot_path = self.models_dir / f"global_model_{self.sba_run_id}_round_{server_round}.json"
        round_path = self.models_dir / f"global_model_round_{server_round}.json"
        latest_path = self.models_dir / "global_model_latest.json"
        snapshot_path.write_text(json.dumps(root, ensure_ascii=False, indent=2))
        round_path.write_text(json.dumps(root, ensure_ascii=False, indent=2))
        latest_path.write_text(json.dumps(root, ensure_ascii=False, indent=2))
        if versioned_latest_path is not None:
            versioned_latest_path.write_text(json.dumps(root, ensure_ascii=False, indent=2))
        logging.info(
            f"SBA_FL_GLOBAL_MODEL_SAVED round={server_round} path={snapshot_path} tensor_count={len(tensors)} latest_path={latest_path} versioned_latest_path={versioned_latest_path}"
        )
        return versioned_latest_path or snapshot_path

    def _write_sba_history_file(self, hist=None, finished=False, duration_seconds=None):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.task_id,
            "sba_fl": dict(self.sba_fl) if hasattr(self.sba_fl, "items") else self.sba_fl,
            "summary": {
                "finished": bool(finished),
                "totalRounds": self.num_rounds,
                "durationSeconds": duration_seconds,
                "globalModelSaved": self.latest_global_model_path is not None,
                "latestModelPath": self.latest_global_model_path,
                "runId": self.sba_run_id,
                "startedAtIso": self.sba_run_started_at,
                "finishedAtIso": self.sba_run_finished_at,
            },
            "rounds": self.sba_round_metrics,
        }
        if hist is not None:
            payload["history"] = {
                "lossesDistributed": getattr(hist, "losses_distributed", []),
                "metricsDistributed": getattr(hist, "metrics_distributed", {}),
            }
        (self.logs_dir / "sba_fl_history.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _save_sba_fl_history(self, hist):
        self._write_sba_history_file(hist=hist, finished=True)

    def start(self):

        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        # self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)

        # New Global Model Version
        # self.server.gl_model_v = self.server.last_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            "S3_bucket": "",  # bucket name
            "Last_GL_Model": "",  # Model Weight File Name
            "FLServer_start": today_time,
            "FLSeReady": True,  # server ready status
            "GL_Model_V": 0 # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            self.sba_run_started_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            fl_start_time = time.time()

            # Run fl server
            self.init_gl_model_registration()

            fl_end_time = time.time() - fl_start_time  # FL end time
            if self.sba_fl:
                self.sba_run_finished_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                self._write_sba_history_file(finished=True, duration_seconds=fl_end_time)

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "gl_model_v": None}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # # # upload global model
            # # if self.model_type == "Tensorflow":
            # #     global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
            # # elif self.model_type =="Pytorch":
            # #     global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
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
