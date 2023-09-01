
import requests


class ServerAPI():
    def __init__(self, task_id):
        self.task_id = task_id
        self.server_manager_address = '10.102.217.157'
        self.server_manager_port = '8000'
        self.performance_address = '10.100.33.100'
        self.performance_port = '8001'

    def put_server_status(self, server_status_json):
        # send server status to server manager
        requests.put(f"http://{self.server_manager_address}:{self.server_manager_port}/FLSe/FLSeUpdate/{self.task_id}", data=server_status_json)

    def put_fl_round_fin(self):
        requests.put(f"http://{self.server_manager_address}:{self.server_manager_port}/FLSe/FLRoundFin/{self.task_id}",
                     params={'FLSeReady': 'false'})

    # send train_result to performance pod
    def put_gl_model_evaluation(self, gl_model_evaluation_json):
        requests.put(f"http://{self.performance_address}:{self.performance_port}/server_perf/gl_model_evaluation/{self.task_id}", data=gl_model_evaluation_json)

    # send server_time_result to performance pod
    def put_server_time_result(self, server_time_result_json):
        requests.put(f"http://{self.performance_address}:{self.performance_port}/server_perf/server_time_result/{self.task_id}", data=server_time_result_json)

