"""
FedOps-WESAD Client Manager — lifecycle daemon for the FL client.

Runs as a FastAPI service on port 8004 alongside client_main.py (:8003).

Responsibilities:
  1. Health-check the FedOps FL server every 10 s
  2. Monitor the Flower client (:8003) every 6 s
  3. Trigger training every 8 s when all conditions are met

Run:
    python client_manager_main.py
"""

import asyncio
import json
import logging
import os
import socket
import sys
import uuid
from datetime import datetime
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI()
today_str = datetime.today().strftime("%Y-%m-%d")


def get_mac_address() -> str:
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i: i + 2] for i in range(0, 12, 2)])


def get_hostname() -> str:
    return socket.gethostname()


class FLTask(BaseModel):
    FL_task_ID:      Optional[str]  = None
    Device_mac:      Optional[str]  = None
    Device_hostname: Optional[str]  = None
    Device_online:   Optional[bool] = None
    Device_training: Optional[bool] = None


class ManagerStatus(BaseModel):
    FL_client:       str  = "localhost:8003"
    server_ST:       str  = "ccl.gachon.ac.kr:40019"
    server:          str  = "ccl.gachon.ac.kr"
    S3_bucket:       str  = "fl-gl-model"
    s3_ready:        bool = False
    GL_Model_V:      int  = 0
    FL_ready:        bool = False
    client_online:   bool = False
    client_training: bool = False
    task_id:         str  = ""
    task_status:     Optional[FLTask] = None
    client_mac:      str  = get_mac_address()
    client_name:     str  = get_hostname()


manager   = ManagerStatus()
inform_SE = f"http://{manager.server_ST}/FLSe/"


@app.get("/info")
def get_manager_info():
    return manager


@app.get("/trainFin")
def fin_train():
    manager.client_training = False
    manager.FL_ready        = False
    _notify_server_closed()
    return manager


@app.get("/trainFail")
def fail_train():
    manager.client_training = False
    manager.FL_ready        = False
    _notify_server_closed()
    return manager


@app.get("/flclient_out")
def flclient_out():
    manager.client_online   = False
    manager.client_training = False
    return manager


def _notify_server_closed():
    try:
        requests.put(
            inform_SE + "FLSeClosed/" + manager.task_id,
            params={"FLSeReady": "false"},
            timeout=5,
        )
    except Exception as exc:
        logger.error("_notify_server_closed error: %s", exc)


def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                await awaitable_func()
            except Exception as exc:
                logger.error("[E] %s: %s", awaitable_func.__name__, exc)
            await asyncio.sleep(0.5)
    return keeping_state


@async_dec
async def health_check():
    status_snapshot = {
        "client_training": manager.client_training,
        "client_online":   manager.client_online,
        "FL_ready":        manager.FL_ready,
    }
    logger.info("health_check — %s", json.dumps(status_snapshot))

    if not manager.FL_ready:
        manager.client_training = False

    if (not manager.client_training) and manager.client_online:
        loop = asyncio.get_event_loop()
        try:
            res = await loop.run_in_executor(
                None,
                requests.get,
                f"http://{manager.server_ST}/FLSe/info/{manager.task_id}/{get_mac_address()}",
            )
            if res.status_code == 200 and res.json()["Server_Status"]["FLSeReady"]:
                manager.FL_ready   = res.json()["Server_Status"]["FLSeReady"]
                manager.GL_Model_V = res.json()["Server_Status"]["GL_Model_V"]
                task_data = res.json()["Server_Status"].get("Task_status")
                manager.task_status = FLTask(**task_data) if task_data else None
        except Exception as exc:
            logger.warning("health_check request failed: %s", exc)

    await asyncio.sleep(10)


@async_dec
async def check_flclient_online():
    logger.info("check_flclient_online")
    if not manager.client_training:
        try:
            loop = asyncio.get_event_loop()
            res_on = await loop.run_in_executor(
                None,
                requests.get,
                f"http://{manager.FL_client}/online",
            )
            if res_on.status_code == 200 and res_on.json().get("client_online"):
                manager.client_online   = res_on.json()["client_online"]
                manager.client_training = res_on.json().get("client_start", False)
                manager.task_id         = res_on.json().get("task_id", "")
        except requests.exceptions.ConnectionError:
            logger.info("Flower client not reachable (not yet started?)")

        try:
            requests.put(
                inform_SE + "RegisterFLTask",
                data=json.dumps({
                    "FL_task_ID":      manager.task_id,
                    "Device_mac":      manager.client_mac,
                    "Device_hostname": manager.client_name,
                    "Device_online":   manager.client_online,
                    "Device_training": manager.client_training,
                }),
                timeout=5,
            )
        except Exception as exc:
            logger.error("RegisterFLTask failed: %s", exc)

    await asyncio.sleep(6)


def _post_request(url, json_data):
    return requests.post(url, json=json_data, timeout=10)


@async_dec
async def start_training():
    if (
        manager.task_status
        and manager.client_online
        and (not manager.client_training)
        and manager.FL_ready
    ):
        logger.info("Triggering FL training ...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                _post_request,
                f"http://{manager.FL_client}/start",
                {"server_ip": manager.server, "client_mac": manager.client_mac},
            )
            manager.client_training = True
        except Exception as exc:
            logger.error("start_training request failed: %s", exc)
    else:
        logger.info(
            "start_training: waiting — online=%s training=%s ready=%s task=%s",
            manager.client_online,
            manager.client_training,
            manager.FL_ready,
            manager.task_status is not None,
        )
    await asyncio.sleep(8)


@app.on_event("startup")
def startup():
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.create_task(check_flclient_online())
    loop.create_task(health_check())
    loop.create_task(start_training())


if __name__ == "__main__":
    uvicorn.run(
        "client_manager_main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        loop="asyncio",
    )
