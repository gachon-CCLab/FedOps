import asyncio
import os, sys
import requests
from pydantic.main import BaseModel
import re
import tensorflow as tf
import logging
import json
from datetime import datetime
from collections import Counter
from keras.utils.np_utils import to_categorical
import itertools


# Log format
handlers_list=[logging.StreamHandler()]
if os.environ["MONITORING"] == '1':
    handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
else:
    pass
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__) 

# server_status Address
inform_SE: str = 'http://ccljhub.gachon.ac.kr:40019/FLSe/'

# client manager address
if len(sys.argv) == 1:
    client_manager_addr = 'http://localhost:8003'
else:
    client_manager_addr = 'http://client-manager:8003'

# FL Client Status class
class FL_client_status(BaseModel):
    FL_client_num: int =  0 # FL client ID
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_round: int = 1 # round
    FL_loss: int = 0 # 
    FL_accuracy: int = 0 
    FL_next_gl_model: int = 0 


# send client name to server_status
def register_client():
    client_name = os.uname()[1]

    res = requests.put(inform_SE + 'RegisterClient', params={'ClientName': client_name})
    if res.status_code == 200:
        client_num = res.json()['client_num']

    return client_num

# load dataset
def load_partition(dataset, FL_client_num):
    # Load the dataset partitions

    # 데이터셋 불러오기
    if dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    if dataset == 'cifar10':
        pass

    else:  # MNIST, FashionMNIST의 모델은 전이학습 모델이므로 3차원으로 설정
        # 28X28 -> 32X32
        # Pad with 2 zeros on left and right hand sides-
        X_train = np.pad(X_train[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
        X_test = np.pad(X_test[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')

        # 배열의 형상을 변경해서 차원 수를 3으로 설정
        # # 전이학습 모델 input값 설정시 차원을 3으로 설정해줘야 함
        X_train = tf.expand_dims(X_train, axis=3, name=None)
        X_test = tf.expand_dims(X_test, axis=3, name=None)
        X_train = tf.repeat(X_train, 3, axis=3)
        X_test = tf.repeat(X_test, 3, axis=3)

    # client_num 값으로 데이터셋 나누기
    (X_train, y_train) = X_train[FL_client_num * 2000:(FL_client_num + 1) * 2000], y_train[FL_client_num * 2000:(FL_client_num + 1) * 2000]
    (X_test, y_test) = X_test[FL_client_num * 1000:(FL_client_num + 1) * 1000], y_test[FL_client_num * 1000:(FL_client_num + 1) * 1000]

    # class 설정
    num_classes = 10

    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # 전처리
    train_features = X_train.astype('float32') / 255.0
    test_features = X_test.astype('float32') / 255.0


    # data check => IID VS Non IID
    # array -> list
    y_list = y_train.tolist()
    y_train_label = list(itertools.chain(*y_list))
    counter = Counter(y_train_label)

    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # check client data(label) => non-iid
    for i in range(num_classes):
        data_check_dict = {"client_num": int(FL_client_num), "label_num": i, "data_size": int(counter[i])}
        data_check_json = json.dumps(data_check_dict)
        logging.info(f'data_check - {data_check_json}')

    return (train_features, train_labels), (test_features, test_labels)

# make local model directory
def make_model_directory():

    # Local Model repository
    if not os.path.isdir('./local_model'):
        os.mkdir('./local_model')
    else:
        pass

# latest local model download
def download_local_model(listdir):
    # mac에서만 시행 (.DS_Store 파일 삭제)
    if '.DS_Store' in listdir:
        i = listdir.index(('.DS_Store'))
        del listdir[i]

    s = listdir[0] 
    p = re.compile(r'\d+')  # Select Number Pattern
    local_list_sorted = sorted(listdir, key=lambda s: int(p.search(s).group()))  # sorting gl_model_version

    local_model_name = local_list_sorted[len(local_list_sorted) - 1]  # select latest gl_model
    model = tf.keras.models.load_model(f'/local_model/{local_model_name}')
    
    logging.info(f'local_model_name: {local_model_name}')

    return model

# check train finish info to client manager
async def notify_fin():

    FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, client_manager_addr + '/trainFin')
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
    future1 = loop.run_in_executor(None, requests.get, client_manager_addr + '/trainFail')
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return FL_client_start