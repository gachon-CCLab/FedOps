from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI

import logging


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class TrainResult(BaseModel):
    fl_task_id: str = ''
    client_mac: str = ''
    round: int = 0
    train_loss: float = 0
    train_acc: float = 0
    execution_time: float = 0
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

class ClientSystem(BaseModel):
    fl_task_id: str = ''
    client_mac: str = ''
    cpu_utilization: float = 0
    system_memory_utilization: float = 0
    process_memory_in_use: float = 0
    process_memory_in_use_size: float = 0
    process_memory_available: float = 0
    process_cpu_threads_in_use: float = 0
    network_traffic: float = 0
    disk_utilization: float = 0
    gpu_utilization: float = 0
    gpu_temp: float = 0
    gpu_time_spent_accessing_memory: float = 0
    gpu_memory_allocated: float = 0
    gpu_power_usage: float = 0
    next_gl_model_v: int = 0

# create App
app = FastAPI()

# Define Class
train_result = TrainResult()
test_result = TestResult()
client_time_result = ClientTimeResult()
client_system = ClientSystem()

@app.put("client_perf/train_result/{task_id}")
def train_result_put(task_id: str, Train: TrainResult):
    global train_result

    train_result.fl_task_id = task_id
    train_result.client_mac = Train.client_mac
    train_result.round = Train.round
    train_result.train_loss = Train.train_loss
    train_result.train_acc = Train.train_acc
    train_result.execution_time = Train.execution_time
    train_result.next_gl_model_v = Train.next_gl_model_v

    logging.info(f'train_result: {train_result}')

    return {"train_result": train_result}


@app.put("client_perf/test_result/{task_id}")
def test_result_put(task_id: str, Test: TestResult):
    global test_result

    test_result.fl_task_id = task_id
    test_result.client_mac = Test.client_mac
    test_result.round = Test.round
    test_result.test_loss = Test.train_loss
    test_result.test_acc = Test.train_acc
    test_result.next_gl_model_v = Test.next_gl_model_v

    logging.info(f'test_result: {test_result}')

    return {"test_result": test_result}

@app.put("client_perf/client_time_result/{task_id}")
def client_time_result_put(task_id: str, Time: ClientTimeResult):
    global client_time_result

    client_time_result.fl_task_id = task_id
    client_time_result.client_mac = Time.client_mac
    client_time_result.operation_time = Time.operation_time
    client_time_result.next_gl_model_v = Time.next_gl_model_v

    logging.info(f'client_time_result: {client_time_result}')

    return {"client_time_result": client_time_result}

@app.put("client_perf/client_system/{task_id}")
def client_system_put(task_id: str, System: ClientSystem):
    global client_system

    client_system.fl_task_id = task_id
    client_system.client_mac = System.client_mac
    client_system.cpu_utilization = System.cpu_utilization
    client_system.system_memory_utilization = System.system_memory_utilization
    client_system.process_memory_in_use = System.process_memory_in_use
    client_system.process_memory_in_use_size = System.process_memory_in_use_size
    client_system.process_memory_available = System.process_memory_available
    client_system.process_cpu_threads_in_use = System.process_cpu_threads_in_use
    client_system.network_traffic = System.network_traffic
    client_system.disk_utilization = System.disk_utilization
    client_system.gpu_utilization = System.gpu_utilization
    client_system.gpu_temp = System.gpu_temp
    client_system.gpu_time_spent_accessing_memory = System.gpu_time_spent_accessing_memory
    client_system.gpu_memory_allocated = System.gpu_memory_allocated
    client_system.gpu_power_usage = System.gpu_power_usage
    client_system.next_gl_model_v = System.next_gl_model_v

    logging.info(f'client_system: {client_system}')

    return {"client_system": client_system}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
