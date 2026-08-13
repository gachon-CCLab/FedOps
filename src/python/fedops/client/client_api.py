#client_api.py
import requests
import sys
import logging, os
from urllib.parse import urlparse

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


def _http_url(environment_name, default):
    value = (os.environ.get(environment_name) or default).strip().rstrip('/')
    parsed = urlparse(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError(f'{environment_name} must be an absolute HTTP URL')
    return value


def _request_timeout():
    try:
        return max(float(os.environ.get('FEDOPS_REQUEST_TIMEOUT_SECONDS', '10')), 0.1)
    except ValueError:
        return 10.0

class ClientMangerAPI():
    def __init__(self,):
        # client manager address
        configured = os.environ.get('FEDOPS_CLIENT_MANAGER_URL', '').strip()
        if configured:
            self.client_manager_addr = _http_url(
                'FEDOPS_CLIENT_MANAGER_URL', 'http://localhost:8004'
            )
        elif len(sys.argv) == 1:
            # When running with shell
            self.client_manager_addr = 'http://localhost:8004'
        else:
            # When running with docker
            self.client_manager_addr = 'http://client-manager:8004'


    def get_info(self):
        client_res = requests.get(
            self.client_manager_addr + '/info/', timeout=_request_timeout()
        )
        return client_res

    def get_client_out(self):
        requests.get(
            self.client_manager_addr + '/flclient_out', timeout=_request_timeout()
        )

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
        self.server_manager_url = _http_url(
            'FEDOPS_SERVER_MANAGER_URL',
            f'http://{self.ccl_address}:{self.server_manager_port}',
        )
        self.client_performance_url = _http_url(
            'FEDOPS_CLIENT_PERFORMANCE_URL',
            f'http://{self.ccl_address}:{self.client_performance_port}',
        )
        self.aggregation_server = os.environ.get(
            'FEDOPS_AGGREGATION_SERVER', ''
        ).strip()

    def get_port(self):
        if self.aggregation_server:
            logger.info('FL_server_IP:port - %s', self.aggregation_server)
            return self.aggregation_server
        # get the FL server IP
        response = requests.get(
            f"{self.server_manager_url}/FLSe/getPort/{self.task_id}",
            timeout=_request_timeout(),
        )
        if response.status_code == 200:
            public_host = os.environ.get('FEDOPS_AGGREGATION_HOST', self.ccl_address)
            FL_server_IP = f"{public_host}:{response.json()['port']}"
            logger.info(f'FL_server_IP:port - {FL_server_IP}')
            return FL_server_IP
        else:
            logger.error(f"Failed to get the port for task {self.task_id} from the server at {self.ccl_address}")

   
        # client_api.py 내부 - 기존 함수만 교체
    def put_cluster_assign(self, client_mac, cluster_id):
        url = f"{self.server_manager_url}/FLSe/cluster/{self.task_id}"

        # None 허용, 값 있으면 int로 강제 캐스팅
        cid = int(cluster_id) if cluster_id is not None else None

        try:
            resp = requests.put(
                url,
                json={"client_mac": client_mac, "cluster_id": cid},
                timeout=_request_timeout(),
            )
            if resp.status_code >= 400:
                logger.warning(
                    f"[cluster-upsert] HTTP {resp.status_code} {resp.text} "
                    f"(task_id={self.task_id}, mac={client_mac}, cluster_id={cid})"
                )
            else:
                # 서버가 {"ok": True}를 주는지 확인 (app.py 그대로면 ok True)
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if not body or body.get("ok") is not True:
                    logger.warning(
                        f"[cluster-upsert] unexpected response: {body} "
                        f"(task_id={self.task_id}, mac={client_mac}, cluster_id={cid})"
                    )
                else:
                    logger.info(
                        f"[cluster-upsert] success (task_id={self.task_id}, mac={client_mac}, cluster_id={cid})"
                    )
        except Exception as e:
            logger.warning(
                f"[cluster-upsert] request failed: {e} "
                f"(task_id={self.task_id}, mac={client_mac}, cluster_id={cid})"
        )
    def put_train_result(self, train_result_json):
        # send train_result to client_performance pod
        requests.put(
            f"{self.client_performance_url}/client_perf/train_result/{self.task_id}",
            data=train_result_json,
            timeout=_request_timeout(),
        )
        
        
    def put_test_result(self, test_result_json):
        requests.put(
            f"{self.client_performance_url}/client_perf/test_result/{self.task_id}",
            data=test_result_json,
            timeout=_request_timeout(),
        )

    def put_client_time_result(self, client_time_result_json):
        requests.put(
            f"{self.client_performance_url}/client_perf/client_time_result/{self.task_id}",
            data=client_time_result_json,
            timeout=_request_timeout(),
        )

    def put_client_system(self, client_system_json):
        requests.put(
            f"{self.client_performance_url}/client_perf/client_system/{self.task_id}",
            data=client_system_json,
            timeout=_request_timeout(),
        )

    def put_client_xai_result(self, xai_result_json):
        requests.put(
            f"{self.client_performance_url}/client_perf/xai_result/{self.task_id}",
            data=xai_result_json,
            timeout=_request_timeout(),
        )
