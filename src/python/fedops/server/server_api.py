#server/server_api.py
import requests
from .connection_config import internal_url


class ServerAPI():
    def __init__(self, task_id):
        self.task_id = task_id
        self.server_manager_url = internal_url('FEDOPS_SERVER_MANAGER_URL')
        self.performance_url = internal_url('FEDOPS_CLIENT_PERFORMANCE_URL')

    def put_server_status(self, server_status_json):
        # send server status to server manager
        requests.put(f"{self.server_manager_url}/FLSe/FLSeUpdate/{self.task_id}", data=server_status_json)

    def put_fl_round_fin(self):
        requests.put(f"{self.server_manager_url}/FLSe/FLRoundFin/{self.task_id}",
                     params={'FLSeReady': 'false'})

    # send train_result to performance pod
    def put_gl_model_evaluation(self, gl_model_evaluation_json):
        requests.put(f"{self.performance_url}/server_perf/gl_model_evaluation/{self.task_id}", data=gl_model_evaluation_json)

    # send server_time_result to performance pod
    def put_server_time_result(self, server_time_result_json):
        requests.put(f"{self.performance_url}/server_perf/server_time_result/{self.task_id}", data=server_time_result_json)

