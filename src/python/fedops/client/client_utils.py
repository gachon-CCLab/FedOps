import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import tensorflow as tf
import logging
import yaml, uuid, socket

from . import client_api

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


# FL Client Status class
class FLClientStatus(BaseModel):
    FL_task_id: str = ''
    FL_client_num: int = 0
    FL_client_mac: str = ''
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_next_gl_model: int = 0


# Get manager data info
class ManagerData(BaseModel):
    server_ip: str
    client_mac: str


def read_config(file_path):
    # Read the YAML configuration file
    config_file_path = file_path
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i + 2] for i in range(0, 12, 2)])


def get_hostname():
    return socket.gethostname()

# make local model directory
def local_model_directory(task_id):
    local_list = []
    # Local Model repository
    if not os.path.isdir(f'./local_model'):
        os.mkdir(f'./local_model')
    else:
        if not os.path.isdir(f'./local_model/{task_id}'):
            os.mkdir(f'./local_model/{task_id}')
        else:
            # check local_model listdir
            local_list = os.listdir(f'./local_model/{task_id}')

    return local_list


# latest local model download
def download_local_model(task_id, listdir):

    pattern = r"([A-Za-z]+)_local_model_V(\d+)\.h5"
    matching_files = [f for f in listdir if re.match(pattern, f)]

    if matching_files:
        latest_local_model_file = sorted(matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
        model_name = re.findall(pattern, latest_local_model_file)[0][0]
        model_path = os.path.join(f"./local_model/{task_id}", latest_local_model_file)

        logging.info(f'local_model_name: {latest_local_model_file}')

        # print(model_path)
        # Load the model
        model = tf.keras.models.load_model(model_path)
    else:
        print("No matching model files found.")

    return model, model_name


# check train finish info to client manager
async def notify_fin():

    FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, client_api.ClientMangerAPI().get_train_fin())
    r = await future2
    logging.info('try notify_fin')
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.error(f'notify_fin error: {r.content}')

    return FL_client_start


# check train fail info to client manager
async def notify_fail():

    logging.info('notify_fail start')

    FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, client_api.ClientMangerAPI().get_train_fail())
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return FL_client_start