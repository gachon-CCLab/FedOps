# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
import itertools
import os, logging, json
import re
import time
from collections import Counter

import tensorflow as tf
from keras.utils.np_utils import to_categorical
import flwr as fl

from functools import partial
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel

import client_data, client_model

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# CPU만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GPU 사용
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# client pod number 추출
pod_name = os.environ['MY_POD_ID'].split('-')
client_num = int(pod_name[2]) # client 번호
print('pod_name: ', pod_name)
print('client_num: ', client_num)


# pytorch로 변경 시 수정 필요
# Client Local Model 생성
def build_model(dataset):

    # 데이터셋 불러오기
    if dataset == 'cifar10':
        model = client_model.model_cnn()

    elif dataset == 'mnist':
        model = client_model.model_ResNet50()
    
    elif dataset == 'fashion_mnist':
        model = client_model.model_VGG16()

    return model


# FL client 상태 확인
app = FastAPI()

# FL Client 상태 class
class FLclient_status(BaseModel):
    FL_client_num: int = client_num # FL client 번호(ID)
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_round: int = 1 # 현재 수행 round
    FL_loss: int = 0 # 성능 loss
    FL_accuracy: int = 0 # 성능 acc
    FL_next_gl_model: int = 0 # 글로벌 모델 버전


status = FLclient_status()

# Define Flower client
class FLClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        # fl_status = fl.common.Status(code=fl.common.Code.OK, message="Success")
        # properties = {"mse": 0.5} # super().get_properties({"mse": 0.5})
        # return fl.common.PropertiesRes(fl_status, properties)
        """Get properties of client."""
        raise Exception("Not implemented")

    # pytorch로 변경 시 수정 필요 (Data format 및 train)
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        global status

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        # num_rounds: int = config["num_rounds"]

        # round 시작 시간
        round_start_time = time.time()

        # local model train
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # round 종료 시간
        round_end_time = time.time() - round_start_time  # 연합학습 종료 시간
        # round_client_operation_time = str(datetime.timedelta(seconds=round_end_time))

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        status.FL_loss = history.history["loss"][len(history.history["loss"])-1]
        status.FL_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
        results = {
            "loss": status.FL_loss,
            "accuracy": status.FL_accuracy,
            "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
            "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
        }

        # Training: model performance by round
        train_result = {"client_num": status.FL_client_num, "round": status.FL_round, "fit_loss": status.FL_loss, "fit_accuracy": status.FL_accuracy,
                        "next_gl_model": status.FL_next_gl_model}
        json_result = json.dumps(train_result)
        logging.info(f'train_performance - {json_result}')
        # print('{"client_num": ' + str(status.FL_client_num) + '{"round": ' + str(status.FL_round) + ', "log": "' + str(json_result) + '"}')

        # Training: model performance by round
        train_time_result = {"client_num": status.FL_client_num, "round": status.FL_round, "next_gl_model": status.FL_next_gl_model, "execution_time": round_end_time}
        json_time_result = json.dumps(train_time_result)
        logging.info(f'train_time - {json_time_result}')

        # save local model
        self.model.save(f'/model/model_V{status.FL_next_gl_model}.h5')

        return parameters_prime, num_examples_train, results

    # pytorch로 변경 시 수정 필요 (Data format)
    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=32, steps=steps)
        num_examples_test = len(self.x_test)

        # Test: model performance by round
        test_result = {"client_num": status.FL_client_num, "round": status.FL_round, "test_loss": test_loss, "test_accuracy": test_accuracy, "next_gl_model": status.FL_next_gl_model}
        json_result = json.dumps(test_result)
        logging.info(f'test - {json_result}')

        # 다음 라운드 수 증가
        status.FL_round += 1

        # print(f'test - client_num: {status.FL_client_num}, round: {status.FL_round}, performance: {json_result}')
        # print('{"client_num": ' + str(status.FL_client_num) + '{"round": ' + str(status.FL_round) + ', "log": "' + str(json_result) + '"}')
        # print('test_loss: ', test_loss, 'test_accuracy: ', test_accuracy)

        return test_loss, num_examples_test, {"accuracy": test_accuracy}


# latest local model download
def download_local_model(listdir):
    # mac에서만 시행 (.DS_Store 파일 삭제)
    if '.DS_Store' in listdir:
        i = listdir.index(('.DS_Store'))
        del listdir[i]

    s = listdir[0]  # 비교 대상(gl_model 지정) => sort를 위함
    p = re.compile(r'\d+')  # 숫자 패턴 추출
    local_list_sorted = sorted(listdir, key=lambda s: int(p.search(s).group()))  # gl model 버전에 따라 정렬

    local_model_name = local_list_sorted[len(local_list_sorted) - 1]  # 최근 gl model 추출
    model = tf.keras.models.load_model(f'/model/{local_model_name}')
    
    # local_model_v = int(local_model_name.split('_')[1])
    logging.info(f'local_model_name: {local_model_name}')

    return model


@app.on_event("startup")
def startup():
    pass

@app.get('/online')
def get_info():
    return status

@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status
    
    # client_manager 주소
    client_res = requests.get('http://localhost:8003/info/')

    # 최신 global model 버전
    latest_gl_model_v = client_res.json()['GL_Model_V']

    # 다음 global model 버전
    status.FL_next_gl_model = latest_gl_model_v + 1

    logging.info('bulid model')

    logging.info('FL start')
    status.FL_client_start = True
    status.FL_server_IP = Server_IP
    background_tasks.add_task(flower_client_start)

    return status


async def flower_client_start():
    logging.info('FL learning ready')
    global status

    # data setting parameter
    all_client_num = 5 # total client number
    dataset = 'fashion_mnist' # dataset
    skewed = True # data partition1: Each client has only one class (or two/three classes)
    skewed_spec = 'skewed_three'
    balanced = False # data partition2: Each client is randomly distributed in different sizes

    # Client Data
    (x_train, y_train), (x_test, y_test) = client_data.data_load(all_client_num, status.FL_client_num, dataset, skewed, skewed_spec, balanced)
    
    # class 설정
    num_classes = 10
    # await asyncio.sleep(30) # data download wait
    logging.info('data loaded')

    # data label check
    if dataset =='cifar10':
        y_list = y_train.tolist()
        y_train_label = list(itertools.chain(*y_list))
        counter = Counter(y_train_label)
    elif dataset=='mnist':
        counter = Counter(y_train)
    elif dataset=='fashion_mnist':
        counter = Counter(y_train)

    for i in range(10):
        data_check_dict = {"client_num": int(status.FL_client_num), "label_num": i, "data_size": int(counter[i])}
        data_check_json = json.dumps(data_check_dict)
        logging.info(f'data_check - {data_check_json}')

    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # model 생성
    model = build_model(dataset)

    # # local_model 유무 확인
    # local_list = os.listdir(f'/model')
    # if not local_list:
    #     logging.info('init local model')
    #     model = build_model()

    # else:
    #     # 최신 local model 다운
    #     logging.info('Latest Local Model download')
    #     model = download_local_model(local_list)

    try:
        loop = asyncio.get_event_loop()
        client = FLClient(model, x_train, train_labels, x_test, test_labels)
        # logging.info(f'fl-server-ip: {status.FL_server_IP}')
        # await asyncio.sleep(23)
        # print('server IP: ', status.FL_server_IP)
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)

        # 라운드 수 초기화
        status.FL_round = 1

        fl_start_time = time.time()  # 연합학습 초기 시작 시간

        await loop.run_in_executor(None, request)  # 연합학습 Client 비동기로 수행

        logging.info('fl learning finished')

        fl_end_time = time.time() - fl_start_time  # 연합학습 종료 시간
        # fl_client_operation_time = str(datetime.timedelta(seconds=fl_end_time))

        client_all_time_result = {"client_num": status.FL_client_num, "operation_time": fl_end_time}
        json_all_time_result = json.dumps(client_all_time_result)
        logging.info(f'client_operation_time - {json_all_time_result}')

        # logging.info(f'fl_client_operation_time: {fl_client_operation_time}')

        # client 객체 및 fl_client_start request 삭제
        del client, request

        # Client learning 완료
        await notify_fin()
        logging.info('FL Client Learning Finish')

    except Exception as e:
        logging.info('[E][PC0002] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False
        raise e

# async def model_save():
    
#     global model, next_gl_model
#     try:
#         model.save('/model/model_V%s.h5'%next_gl_model)
#         await notify_fin()
#         model=None
#     except Exception as e:
#         logging.info('[E][PC0003] learning', e)
#         status.FL_client_fail = True
#         await notify_fail()
#         status.FL_client_fail = False

#     return status

# client manager에서 train finish 정보 확인
async def notify_fin():
    global status

    status.FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFin')
    r = await future2
    logging.info('try notify_fin')
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.error(f'notify_fin error: {r.content}')
    return status


# client manager에서 train fail 정보 확인
async def notify_fail():
    global status

    logging.info('notify_fail start')

    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return status


if __name__ == "__main__":

    try:
        # client api 생성 => client manager와 통신하기 위함
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)
        
    finally:
        # FL client out
        requests.get('http://localhost:8003/flclient_out')
        logging.info('%s client close'%client_num)
