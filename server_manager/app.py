from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Optional
import json
import logging
import datetime
from typing import List

from utils import server_operator

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

FL_task_list = []

# Create a dictionary to store the statuses of the tasks
fl_server_status = {}

# Create a dictionary to store the ServerStatus objects for each task
FLSe_dict = {}


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
    Task_status: FLTask = None

    def to_json(self):
        return jsonable_encoder(self)


class StartingTaskData(BaseModel):
    task_id: str
    devices: List[str]


# create App
app = FastAPI()

# create Object
FLSe = ServerStatus()


# Create a function to get or create a ServerStatus object for a task
def get_or_create_FLSe(task_id: str):
    if task_id not in FLSe_dict:
        FLSe_dict[task_id] = ServerStatus()
    return FLSe_dict[task_id]


@app.get("/FLSe/info/{task_id}/{device_mac}")
def read_status(task_id: str, device_mac: str):
    try:
        global FLSe, FL_task_list
        FLSe = get_or_create_FLSe(task_id)

        # Filter the FL_task_list based on task_id and device_mac
        matching_tasks = [task for task in FL_task_list if task.FL_task_ID == task_id and task.Device_mac == device_mac]

        if not matching_tasks:
            server_status_result = {
                "Play_datetime": FLSe.Server_manager_start,
                "FLSeReady": FLSe.FLSeReady,
                "GL_Model_V": FLSe.GL_Model_V,
                "Task_Status": None
            }
            logging.info(f'server_status - {server_status_result}')
            FLSe.Task_status = None
        else:
            # Get the first matching task
            matching_task = matching_tasks[0]

            # Convert the matching_task to a JSON-serializable format
            matching_task_json = jsonable_encoder(matching_task)

            server_status_result = {
                "Play_datetime": FLSe.Server_manager_start,
                "FLSeReady": FLSe.FLSeReady,
                "GL_Model_V": FLSe.GL_Model_V,
                "Task_Status": matching_task_json
            }

            logging.info(f'server_status - {server_status_result}')

            FLSe.Task_status = matching_task

        return JSONResponse(content={"Server_Status": FLSe.to_json()})
    except Exception as e:
        logging.error(f"Error in read_status: {str(e)}")
        return {"error": str(e)}



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
    global FL_task_list
    update_or_append_task(task)

    logging.info(f'registered_fl_task_list: {task}')
    logging.info(f'registered_fl_task_lists: {FL_task_list}')

    return FL_task_list


@app.get("/FLSe/GetFLTask/{task_id}")
def get_fl_task(task_id: str):
    global FL_task_list
    matching_tasks = []

    # Find the matching FLTask instances
    for task in FL_task_list:
        if task.FL_task_ID == task_id:
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
def start_task(task_data: StartingTaskData, background_tasks: BackgroundTasks):
    # Start the task and create a background task to manage its status
    background_tasks.add_task(server_operator.create_fl_server, task_data.task_id, fl_server_status)

    return {"status": "Task started."}


@app.get("/FLSe/status/{task_id}")
def get_fl_server_status(task_id: str):
    status = fl_server_status.get(task_id)
    if status:
        return {"task_id": task_id, "status": status}
    else:
        return {"error": f"No task with id {task_id} found"}


@app.put("/FLSe/FLSeUpdate/{task_id}")
def update_status(task_id: str, Se: ServerStatus):
    global FLSe_dict
    FLSe = get_or_create_FLSe(task_id)
    FLSe.S3_bucket = Se.S3_bucket
    FLSe.Latest_GL_Model = Se.Latest_GL_Model
    FLSe.Server_manager_start = Se.Server_manager_start
    FLSe.FLSeReady = Se.FLSeReady
    FLSe.GL_Model_V = Se.GL_Model_V
    FLSe.Task_status = Se.Task_status
    return {"Server_Status": FLSe}


@app.put("/FLSe/FLRoundFin/{task_id}")
def update_ready(task_id: str, FLSeReady: bool):
    global FLSe_dict
    FLSe = get_or_create_FLSe(task_id)
    FLSe.FLSeReady = FLSeReady
    if FLSeReady==False:
        FLSe.GL_Model_V += 1
    return {"Server_Status": FLSe}


@app.put("/FLSe/FLSeClosed/{task_id}")
def server_closed(task_id: str, FLSeReady: bool):
    global FLSe, FL_task_list, fl_server_status, FLSe_dict
    FLSe = get_or_create_FLSe(task_id)

    # Clear FL_task_list for matching task_id
    FL_task_list = [task for task in FL_task_list if task.FL_task_ID != task_id]

    # Clear fl_server_status for matching task_id
    if task_id in fl_server_status:
        del fl_server_status[task_id]

    # Clear FLSe_dict for matching task_id
    # if task_id in FLSe_dict:
    #     del FLSe_dict[task_id]

    print('server closed')
    FLSe.FLSeReady = FLSeReady
    return {"Server_Status": FLSe}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
