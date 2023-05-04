from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Optional
import json, logging
import datetime

from utils import server_operator

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

FL_task_list = []
class FLTask(BaseModel):
    FL_task_ID: str = ''
    Device_mac: str = ''
    Device_hostname: str = ''
    Device_online: bool = False
    Device_training: bool = False
    # Device_time: str = ''

# Server Status Object
class ServerStatus(BaseModel):

    S3_bucket: str = 'fl-gl-model'
    Latest_GL_Model: str = ''  # 모델 가중치 파일 이름
    Server_manager_start: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    FLSeReady: bool = False
    GL_Model_V: int = 0  # 모델버전

class StartingTaskData(BaseModel):
    task_id: str
    devices: List[str]

# create App
app = FastAPI()

# create Object
FLSe = ServerStatus()

@app.get("/FLSe/info")
def read_status():
    global FLSe

    # server_status_result = {"S3_bucket": FLSe.S3_bucket, "Latest_GL_Model": FLSe.Latest_GL_Model, "Play_datetime": FLSe.Play_datetime,
    #                         "FLSeReady": FLSe.FLSeReady, "GL_Model_V": FLSe.GL_Model_V}
    server_status_result = {"Play_datetime": FLSe.Server_manager_start, "FLSeReady": FLSe.FLSeReady, "GL_Model_V": FLSe.GL_Model_V}
    json_server_status_result = json.dumps(server_status_result)
    logging.info(f'server_status - {json_server_status_result}')
    # print(FLSe)
    return {"Server_Status": FLSe}


def update_or_append_task(new_task):
    global FL_task_list
    found = False
    for task in FL_task_list:
        if task.Device_mac == new_task.Device_mac:
            task.Device_hostname = new_task.Device_hostname
            task.Device_online = new_task.Device_online
            task.Device_training = new_task.Device_training
            found = True
            break

    if not found:
        FL_task_list.append(new_task)


@app.put("/FLSe/RegisterFLTask")
def register_fl_task(task: FLTask):
    global FLSe, FL_task_list
    # task.Device_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    update_or_append_task(task)

    logging.info(f'registered_fl_task_list: {task}')
    logging.info(f'registered_fl_task_lists: {FL_task_list}')

    return FL_task_list


@app.get("/FLSe/GetFLTask/{FL_task_ID}")
def get_fl_task(FL_task_ID: str):
    global FL_task_list
    matching_tasks = []

    # Find the matching FLTask instances
    for task in FL_task_list:
        if task.FL_task_ID == FL_task_ID:
            matching_tasks.append(task)

    # Convert the matching FLTask instances to a JSON-compatible list
    tasks_json = jsonable_encoder(matching_tasks)

    # If no matching instances are found, return an error message
    if not tasks_json:
        return {"error": "No matching FLTask found for the provided FL_task_ID"}

    return tasks_json

# {
#   "task_id": "some_task_id",
#   "devices": [
#     "device_mac_1",
#     "device_mac_2",
#     "device_mac_3"
#   ]
# }
@app.post("/FLSe/startTask")
def start_task(task_data: StartingTaskData):
    # print(task_data.task_id)
    # print(task_data.devices)


@app.put("/FLSe/FLSeUpdate")
def update_status(Se: ServerStatus):
    global FLSe
    FLSe = Se
    return {"Server_Status": FLSe}

@app.put("/FLSe/FLRoundFin")
def update_ready(FLSeReady: bool):
    global FLSe
    FLSe.FLSeReady = FLSeReady
    if FLSeReady==False:
        FLSe.GL_Model_V += 1
    return {"Server_Status": FLSe}


@app.put("/FLSe/FLSeClosed")
def server_closed(FLSeReady: bool):
    global FLSe
    print('server closed')
    FLSe.FLSeReady = FLSeReady
    return {"Server_Status": FLSe}



if __name__ == "__main__":    

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
