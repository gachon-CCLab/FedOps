from pydantic.main import BaseModel
import logging
import requests
import uvicorn
from fastapi import FastAPI
import asyncio
import json
from datetime import datetime
import requests
import os, sys

handlers_list=[logging.StreamHandler()]
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
today= datetime.today()
today_str = today.strftime('%Y-%m-%d')

class manager_status(BaseModel):
    global today_str

    FL_client: str = ''
    if len(sys.argv) == 1: 
        FL_client = 'localhost:8002'
    else: 
        FL_client = 'fl-client:8002'
    FL_server_ST: str = 'ccljhub.gachon.ac.kr:40019'
    FL_server: str = 'ccljhub.gachon.ac.kr:40018' 
    S3_bucket: str = 'fl-gl-model'
    s3_ready: bool = False  
    FL_client_num: int = 0
    GL_Model_V: int = 0  # model version
    FL_ready: bool = False  

    FL_client_online: bool = False  # flower client online
    FL_learning: bool = False  # flower client learning

    # infer_online: bool = False  # infer online?
    # infer_running: bool = False  # inference server 작동중
    # infer_updating:bool = False #i inference server 업데이트중
    # infer_ready: bool = False  # 모델이 준비되어있음 infer update 필요


manager = manager_status()


@app.on_event("startup")
def startup():
    ##### S0 #####
    
    # get_server_info()

    # create_task를 해야 여러 코루틴을 동시에 실행
    # asyncio.create_task(pull_model())
    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    # 전역변수값을 보고 상태를 유지하려고 합니다.
    # 이런식으로 짠 이유는 개발과정에서 각 구성요소의 상태가 불안정할수 있기 때문으로
    # manager가 일정주기로 상태를 확인하고 또는 명령에 대한 반환값을 가지고 정보를 갱신합니다
    loop.create_task(check_flclient_online())
    loop.create_task(health_check())
    loop.create_task(start_training())

    # 코루틴이 여러개일 경우, asyncio.gather을 먼저 이용 (순서대로 스케쥴링 된다.)
    # loop.run_until_complete(asyncio.gather(health_check(), check_flclient_online(), start_training()))


# fl server occured error
def fl_server_closed():
    global manager

    inform_SE: str = f'http://{manager.FL_server_ST}/FLSe/'
    try: 
        requests.put(inform_SE+'FLSeClosed', params={'FLSeReady': 'false'})
        logging.info('server status FLSeReady => False')
    except Exception as e:
        logging.error(f'fl_server_closed error: {e}')


@app.get("/trainFin")
def fin_train():
    global manager
    logging.info('fin')
    manager.FL_learning = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get("/trainFail")
def fail_train():
    global manager
    logging.info('Fail')
    manager.FL_learning = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get('/info')
def get_manager_info():
    return manager

@app.get('/flclient_out')
def flclient_out():
    manager.FL_client_online = False
    manager.FL_learning = False
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
            await asyncio.sleep(1)

    return keeping_state


# check Server Status
@async_dec
async def health_check():
    global manager

    health_check_result = {"client_num": manager.FL_client_num, "FL_learning": manager.FL_learning, "FL_client_online": manager.FL_client_online, "FL_ready": manager.FL_ready}
    json_result = json.dumps(health_check_result)
    logging.info(f'health_check - {json_result}')

    # if Server is Off, Client Local Learning = False
    if manager.FL_ready == False:
        manager.FL_learning == False

    if (manager.FL_learning == False) and (manager.FL_client_online == True):
        loop = asyncio.get_event_loop()
        # raise
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_server_ST + '/FLSe/info'))
        if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
            manager.FL_ready = res.json()['Server_Status']['FLSeReady']
            manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']

        elif (res.status_code != 200):
            # manager.FL_client_online = False
            logging.error('FL_server_ST offline')
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
    if (manager.FL_learning==False):
        try:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/online'))
            if (res.status_code == 200) and (res.json()['FL_client_online']):
                manager.FL_client_online = res.json()['FL_client_online']
                manager.FL_learning = res.json()['FL_client_start']
                manager.FL_client_num = res.json()['FL_client_num']
                print('FL_client_online: ', manager.FL_client_online, ' FL_client_num: ',manager.FL_client_num)
                logging.info('FL_client online')

            else:
                logging.info('FL_client offline')
                pass
        except requests.exceptions.ConnectionError:
            logging.info('FL_client offline')
            pass
    else:
        pass
    
    await asyncio.sleep(12)
    return manager

# make trigger for client fl start
@async_dec
async def start_training():
    global manager
    # logging.info(f'start_training - FL Client Learning: {manager.FL_learning}')
    # logging.info(f'start_training - FL Client Online: {manager.FL_client_online}')
    # logging.info(f'start_training - FL Server Status: {manager.FL_ready}')

    if (manager.FL_client_online == True) and (manager.FL_learning == False) and (manager.FL_ready == True):
        logging.info('start training')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/start/'+manager.FL_server))
        manager.FL_learning = True
        logging.info(f'client_start code: {res.status_code}')
        if (res.status_code == 200) and (res.json()['FL_client_start']):
            logging.info('flclient learning')
            
        elif (res.status_code != 200):
            manager.FL_client_online = False
            logging.info('flclient offline')
        else:
            pass
    else:
        # await asyncio.sleep(11)
        pass

    await asyncio.sleep(13)
    return manager

# get init server info
# def get_server_info():
#     global manager
#     try:
#         logging.info('get_server_info')
#         logging.info(f'get_server_info() FL_ready: {manager.FL_ready}')
#         res = requests.get('http://' + manager.FL_server_ST + '/FLSe/info')
#         manager.S3_bucket = res.json()['Server_Status']['S3_bucket']
#         manager.s3_ready = True
#         # manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']
#         # manager.FL_ready = res.json()['Server_Status']['FLSeReady']
#     except Exception as e:
#         raise e
#     return manager


if __name__ == "__main__":
    # asyncio.run(training())
    uvicorn.run("client_manager:app", host='0.0.0.0', port=8003, reload=True)
