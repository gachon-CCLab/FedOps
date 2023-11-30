import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import logging
import uuid, socket
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
    task_id: str = ''
    client_num: int = 0
    client_mac: str = ''
    client_name: str = ''
    client_online: bool = True
    client_start: bool = False
    client_fail: bool = False
    server_IP: str = None # FL server IP
    next_gl_model: int = 0


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
        
    if not os.path.isdir(f'./local_model/{task_id}'):
        os.mkdir(f'./local_model/{task_id}')
        local_list = os.listdir(f'./local_model/{task_id}')
        
    else:
        local_list = os.listdir(f'./local_model/{task_id}')

    return local_list


# latest local model download
def download_local_model(model_type, task_id, listdir, model=None):

    pattern = r"([A-Za-z]+)_local_model_V(\d+)\.(h5|pth)"
    matching_files = [f for f in listdir if re.match(pattern, f)]

    if matching_files:
        latest_local_model_file = sorted(matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
        # model_name = re.findall(pattern, latest_local_model_file)[0][0]
        model_extension = re.findall(pattern, latest_local_model_file)[0][2]
        model_path = os.path.join(f"./local_model/{task_id}/", latest_local_model_file)

        logging.info(f'downloaded local_model_name: {latest_local_model_file}')
        
        # Initialize downloaded model
        # downloaded_model = None

        if model_type == "Tensorflow" and model_extension == "h5":
            # Load TensorFlow Keras model
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)

        elif model_type == "Pytorch" and model_extension == "pth":
            import torch
            model.load_state_dict(torch.load(model_path))

    else:
        print("No matching model files found.")

    return model


# check train finish info to client manager
async def notify_fin():
    logging.info('try notify_fin')
    FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, client_api.ClientMangerAPI().get_train_fin())
    r = await future2
    
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