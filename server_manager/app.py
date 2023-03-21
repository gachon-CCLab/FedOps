from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
import json, logging


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Server Status Object
class ServerStatus(BaseModel):

    S3_bucket: str = 'fl-gl-model'
    Latest_GL_Model: str = '' # 모델 가중치 파일 이름
    Play_datetime: str = ''
    FLSeReady: bool = False
    GL_Model_V: int = 0 #모델버전


# init client_list object
client_list= []

# create App
app = FastAPI()

# create Object
FLSe = ServerStatus()

@app.get("/FLSe/info")
def read_status():
    global FLSe

    server_status_result = {"S3_bucket": FLSe.S3_bucket, "Latest_GL_Model": FLSe.Latest_GL_Model, "Play_datetime": FLSe.Play_datetime,
                            "FLSeReady": FLSe.FLSeReady, "GL_Model_V": FLSe.GL_Model_V}
    json_server_status_result = json.dumps(server_status_result)
    logging.info(f'server_status - {json_server_status_result}')
    # print(FLSe)
    return {"Server_Status": FLSe}


@app.put("/FLSe/RegisterClient")
def register_client(ClientName: str):
    global FLSe
    if ClientName in client_list:
        client_num = client_list.index(ClientName)
    else: 
        client_list.append(ClientName)
        client_num = client_list.index(ClientName)

    logging.info(f'registered_client_list: {client_list}')

    return {'client_num':client_num}


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
