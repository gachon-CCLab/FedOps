from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI

import logging, os
import pymongo


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class TrainResult(BaseModel):
    fl_task_id: str = ''
    client_mac: str = ''
    round: int = 0
    train_loss: float = 0
    train_acc: float = 0
    train_time: float = 0
    next_gl_model_v: int = 0

class TestResult(BaseModel):
    fl_task_id: str = ''
    client_mac: str = ''
    round: int = 0
    test_loss: float = 0
    test_acc: float = 0
    next_gl_model_v: int = 0

class ClientTimeResult(BaseModel):
    fl_task_id: str = ''
    client_mac: str = ''
    operation_time: float = 0
    next_gl_model_v: int = 0

class ClientBasicSystem(BaseModel):
    network_sent: float = 0
    network_recv: float = 0
    disk: float = 0
    runtime: float = 0
    memory_rssMB: float = 0
    memory_availableMB: float = 0
    cpu: float = 0
    cpu_threads: float = 0
    memory: float = 0
    memory_percent: float = 0
    timestamp: float = 0
    fl_task_id: str = ''
    client_mac: str = ''
    next_gl_model_v: int = 0



# class ClientGpuSystem(BaseModel):


# create App
app = FastAPI()

# Define Class
train_result = TrainResult()
test_result = TestResult()
client_time_result = ClientTimeResult()
client_basic_system = ClientBasicSystem()

# MogoDB
MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DATABASE = os.environ["MONGODB_DATABASE"]

# Connect MongoDB
client = pymongo.MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]

@app.put("/client_perf/train_result/{task_id}")
def train_result_put(task_id: str, Train: TrainResult):
    global train_result, db

    train_result.fl_task_id = task_id
    train_result.client_mac = Train.client_mac
    train_result.round = Train.round
    train_result.train_loss = Train.train_loss
    train_result.train_acc = Train.train_acc
    train_result.train_time = Train.train_time
    train_result.next_gl_model_v = Train.next_gl_model_v

    logging.info(f'train_result: {train_result}')

    collection = db["fl_client_train_result_log"]

    # input train_result data
    document = {
        "fl_task_id": task_id,
        "client_mac": train_result.client_mac,
        "round": train_result.round,
        "train_loss": train_result.train_loss,
        "train_acc": train_result.train_acc,
        "train_time": train_result.train_time,
        "next_gl_model_v": train_result.next_gl_model_v
    }

    collection.insert_one(document)

    return {"train_result": train_result}


@app.put("/client_perf/test_result/{task_id}")
def test_result_put(task_id: str, Test: TestResult):
    global test_result, db

    test_result.fl_task_id = task_id
    test_result.client_mac = Test.client_mac
    test_result.round = Test.round
    test_result.test_loss = Test.test_loss
    test_result.test_acc = Test.test_acc
    test_result.next_gl_model_v = Test.next_gl_model_v

    logging.info(f'test_result: {test_result}')

    collection = db["fl_client_test_train_log"]

    # input test_result data
    document = {
        "fl_task_id": task_id,
        "client_mac": test_result.client_mac,
        "round": round,
        "test_loss": test_result.test_loss,
        "test_acc": test_result.test_acc,
        "next_gl_model_v": test_result.next_gl_model_v
    }

    collection.insert_one(document)

    return {"test_result": test_result}

@app.put("/client_perf/client_time_result/{task_id}")
def client_time_result_put(task_id: str, Time: ClientTimeResult):
    global client_time_result, db

    client_time_result.fl_task_id = task_id
    client_time_result.client_mac = Time.client_mac
    client_time_result.operation_time = Time.operation_time
    client_time_result.next_gl_model_v = Time.next_gl_model_v

    logging.info(f'client_time_result: {client_time_result}')

    collection = db["fl_client_time_result_log"]

    # input client_time_result data
    document = {
        "fl_task_id": task_id,
        "client_mac": client_time_result.client_mac,
        "operation_time": client_time_result.operation_time,
        "next_gl_model_v": client_time_result.next_gl_model_v
    }

    collection.insert_one(document)

    return {"client_time_result": client_time_result}

@app.put("/client_perf/client_system/{task_id}")
def client_basic_system_put(task_id: str, System: ClientBasicSystem):
    global client_basic_system, db

    client_basic_system.network_sent = System.network_sent
    client_basic_system.network_recv = System.network_recv
    client_basic_system.disk = System.disk
    client_basic_system.runtime = System.runtime
    client_basic_system.memory_rssMB = System.memory_rssMB
    client_basic_system.memory_availableMB = System.memory_availableMB
    client_basic_system.cpu = System.cpu
    client_basic_system.cpu_threads = System.cpu_threads
    client_basic_system.memory = System.memory
    client_basic_system.memory_percent = System.memory_percent
    client_basic_system.timestamp = System.timestamp
    client_basic_system.fl_task_id = task_id
    client_basic_system.client_mac = System.client_mac
    client_basic_system.next_gl_model_v = System.next_gl_model_v

    logging.info(f'client_basic_system: {client_basic_system}')

    collection = db["client_basic_system"]

    # input client_basic_system data
    document = {
        "fl_task_id": task_id,
        "client_mac": client_basic_system.client_mac,
        "network_sent": client_basic_system.network_sent,
        "network_recv": client_basic_system.network_recv,
        "disk": client_basic_system.disk,
        "runtime": client_basic_system.runtime,
        "memory_rssMB": client_basic_system.memory_rssMB,
        "memory_availableMB": client_basic_system.memory_availableMB,
        "cpu": client_basic_system.cpu,
        "cpu_threads": client_basic_system.cpu_threads,
        "memory": client_basic_system.memory,
        "memory_percent": client_basic_system.memory_percent,
        "timestamp": client_basic_system.timestamp,
        "next_gl_model_v": client_basic_system.next_gl_model_v
    }

    collection.insert_one(document)


    return {"client_system": client_basic_system}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
