import requests
import sys
import logging, os

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

class ClientMangerAPI():
    def __init__(self,):
        # client manager address
        if len(sys.argv) == 1:
            # When running with shell
            self.client_manager_addr = 'http://localhost:8003'
        else:
            # When running with docker
            self.client_manager_addr = 'http://client-manager:8003'


    def get_info(self):
        client_res = requests.get(self.client_manager_addr + '/info/')
        return client_res

    def get_client_out(self):
        requests.get(self.client_manager_addr + '/flclient_out')

    def get_train_fin(self):
        train_fin = f"{self.client_manager_addr}/trainFin"
        return train_fin

    def get_train_fail(self):
        train_fail = f"{self.client_manager_addr}/trainFail"
        return train_fail


class ClientServerAPI():
    def __init__(self, task_id):
        self.task_id = task_id
        self.ccl_address = 'ccl.gachon.ac.kr'
        self.server_manager_port = '40019'
        self.client_performance_port = '40015'

    def get_port(self):
        # get the FL server IP
        response = requests.get(f"http://{self.ccl_address}:{self.server_manager_port}/FLSe/getPort/{self.task_id}")
        if response.status_code == 200:
            FL_server_IP = f"{self.ccl_address}:{response.json()['port']}"
            logger.info(f'FL_server_IP:port - {FL_server_IP}')
            return FL_server_IP
        else:
            logger.error(f"Failed to get the port for task {self.task_id} from the server at {self.ccl_address}")

    def put_train_result(self, train_result_json):
        # send train_result to client_performance pod
        requests.put(f"http://{self.ccl_address}:{self.client_performance_port}/client_perf/train_result/{self.task_id}", data=train_result_json)

    def put_test_result(self, test_result_json):
        requests.put(f"http://{self.ccl_address}:{self.client_performance_port}/client_perf/test_result/{self.task_id}", data=test_result_json)

    def put_client_time_result(self, client_time_result_json):
        requests.put(f"http://{self.ccl_address}:{self.client_performance_port}/client_perf/client_time_result/{self.task_id}", data=client_time_result_json)

    def put_client_system(self, client_system_json):
        requests.put(f"http://{self.ccl_address}:{self.client_performance_port}/client_perf/client_system/{self.task_id}", data=client_system_json)


