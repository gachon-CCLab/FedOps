
#client_manager_main.py

from pydantic.main import BaseModel
import logging
import uvicorn
from fastapi import FastAPI
import asyncio
import json
from datetime import datetime
import requests
import os
import sys
import yaml
import uuid
import socket
from typing import Optional

handlers_list = [logging.StreamHandler()]
if "MONITORING" in os.environ:
    if os.environ["MONITORING"] == '1':
        handlers_list.append(logging.FileHandler('./fedops/client_manager.log'))
    else:
        pass
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)
app = FastAPI()

# 날짜를 폴더로 설정
global today_str
today = datetime.today()
today_str = today.strftime('%Y-%m-%d')

global inform_SE

def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i + 2] for i in range(0, 12, 2)])


def get_hostname():
    return socket.gethostname()


class FLTask(BaseModel):
    FL_task_ID: Optional[str] = None
    Device_mac: Optional[str] = None
    Device_hostname: Optional[str] = None
    Device_online: Optional[bool] = None
    Device_training: Optional[bool] = None


class manager_status(BaseModel):
    global today_str, inform_SE

    # FL_client: str = '0.0.0.0:8003'
    if len(sys.argv) == 1:
        FL_client : str = 'localhost:8003'
    else:
        FL_client = 'fl-client:8003'
    server_ST: str = 'ccl.gachon.ac.kr:40019'
    server: str = 'ccl.gachon.ac.kr'
    S3_bucket: str = 'fl-gl-model'
    s3_ready: bool = False
    GL_Model_V: int = 0  # model version
    FL_ready: bool = False

    client_online: bool = False  # flower client online
    client_training: bool = False  # flower client learning

    task_id: str = ''
    task_status: FLTask = None

    client_mac: str = get_mac_address()
    client_name: str = get_hostname()

    inform_SE = f'http://{server_ST}/FLSe/'


manager = manager_status()


@app.on_event("startup")
def startup():
    ##### S0 #####
    
    # get_server_info()

    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.create_task(check_flclient_online())
    loop.create_task(health_check())
    # loop.create_task(register_client())
    loop.create_task(start_training())



# fl server occured error
def fl_server_closed():
    global manager
    try: 
        requests.put(inform_SE + 'FLSeClosed/' + manager.task_id, params={'FLSeReady': 'false'})
        logging.info('server status FLSeReady => False')
    except Exception as e:
        logging.error(f'fl_server_closed error: {e}')


@app.get("/trainFin")
def fin_train():
    global manager
    logging.info('fin')
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get("/trainFail")
def fail_train():
    global manager
    logging.info('Fail')
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get('/info')
def get_manager_info():
    return manager


@app.get('/flclient_out')
def flclient_out():
    manager.client_online = False
    manager.client_training = False
    return manager


def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                # logging.debug(str(awaitable_func.__name__) + '함수 시작')
                # print(awaitable_func.__name__, '함수 시작')
                await awaitable_func()
                # logging.debug(str(awaitable_func.__name__) + '_함수 종료')
            except Exception as e:
                # logging.info('[E]' , awaitable_func.__name__, e)
                logging.error('[E]' + str(awaitable_func.__name__)+ ': ' + str(e))
            await asyncio.sleep(0.5)

    return keeping_state


# send client name to server_status
# @async_dec
# async def register_client():
#     global manager, inform_SE

#     res = requests.put(inform_SE + 'RegisterFLTask',
#                        data=json.dumps({
#                            'FL_task_ID': manager.task_id,
#                            'Device_mac': manager.client_mac,
#                            'Device_hostname': manager.client_name,
#                            'Device_online': manager.client_online,
#                            'Device_training': manager.client_training,
#                        }))

#     if res.status_code == 200:
#         pass
#     else:
#         logging.error('FLSe/RegisterFLTask: server_ST offline')
#         pass

#     await asyncio.sleep(14)
#     return manager


# check Server Status
@async_dec
async def health_check():
    global manager

    health_check_result = {
        "client_training": manager.client_training,
        "client_online": manager.client_online,
        "FL_ready": manager.FL_ready
    }
    json_result = json.dumps(health_check_result)
    logging.info(f'health_check - {json_result}')

    # If Server is Off, Client Local Learning = False
    if not manager.FL_ready:
        manager.client_training = False

    if (not manager.client_training) and manager.client_online:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None, requests.get, (
                    'http://' + manager.server_ST + '/FLSe/info/' + manager.task_id + '/' + get_mac_address()
            )
        )
        if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
            manager.FL_ready = res.json()['Server_Status']['FLSeReady']
            manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']

            # Update manager.FL_task_status based on the server's response
            task_status_data = res.json()['Server_Status']['Task_status']
            logging.info(f'task_status_data - {task_status_data}')
            if task_status_data is not None:
                manager.task_status = FLTask(**task_status_data)
            else:
                manager.task_status = None

        elif (res.status_code != 200):
            # manager.FL_client_online = False
            logging.error('FLSe/info: ' + str(res.status_code) + ' FL_server_ST offline')
            # exit(0)
        else:
            pass
    else:
        pass
    await asyncio.sleep(10)
    return manager


# check client status
@async_dec
async def check_flclient_online():
    global manager
    logging.info('Check client online info')
    if not manager.client_training:
        try:
            loop = asyncio.get_event_loop()
            res_on = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/online'))
            if (res_on.status_code == 200) and (res_on.json()['client_online']):
                manager.client_online = res_on.json()['client_online']
                manager.client_training = res_on.json()['client_start']
                manager.task_id = res_on.json()['task_id']
                logging.info('client_online')

            else:
                logging.info('client offline')
                pass
        except requests.exceptions.ConnectionError:
            logging.info('client offline')
            pass
        
        res_task = requests.put(inform_SE + 'RegisterFLTask',
                       data=json.dumps({
                           'FL_task_ID': manager.task_id,
                           'Device_mac': manager.client_mac,
                           'Device_hostname': manager.client_name,
                           'Device_online': manager.client_online,
                           'Device_training': manager.client_training,
                       }))

        if res_task.status_code == 200:
            pass
        else:
            logging.error('FLSe/RegisterFLTask: server_ST offline')
            pass
        
    else:
        pass
    
    await asyncio.sleep(6)
    return manager


# Helper function to make the POST request
def post_request(url, json_data):
    return requests.post(url, json=json_data)


# make trigger for client fl start
@async_dec
async def start_training():
    global manager
    # logging.info(f'start_training - FL Client Learning: {manager.FL_learning}')
    # logging.info(f'start_training - FL Client Online: {manager.FL_client_online}')
    # logging.info(f'start_training - FL Server Status: {manager.FL_ready}')

    # Check if the FL_task_status is not None
    if manager.task_status:
        if manager.client_online and (not manager.client_training) and manager.FL_ready:
            logging.info('start training')
            loop = asyncio.get_event_loop()
            # Use the helper function with run_in_executor
            res = await loop.run_in_executor(None, post_request,
                                             'http://' + manager.FL_client + '/start', {"server_ip": manager.server, "client_mac": manager.client_mac})

            manager.client_training = True
            logging.info(f'client_start code: {res.status_code}')
            if (res.status_code == 200) and (res.json()['FL_client_start']):
                logging.info('flclient learning')

            elif res.status_code != 200:
                manager.client_online = False
                logging.info('flclient offline')
            else:
                pass
        else:
            # await asyncio.sleep(11)
            pass
    else:
        logging.info("FL_task_status is None")

    await asyncio.sleep(8)
    return manager


if __name__ == "__main__":
    # asyncio.run(training())
    uvicorn.run("client_manager_main:app", host='0.0.0.0', port=8004, reload=True, loop="asyncio")