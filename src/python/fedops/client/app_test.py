import logging, json
import time

from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn

import client_utils
import client_task
import client_fl
import client_wandb
import client_api
from datetime import datetime

# read config.yaml file
config = client_utils.read_config()

# FL task ID
task_id = config['task']['name']

# Data name
dataset = config['data']['name']

# Label count
label_count = config['data']['label_count']

# Train validation split
validation_split = config['data']['validation_split']

# API that checks FL client status
app = FastAPI()

# FL Client Status Object
status = client_utils.FLClientStatus()

@app.get('/online')
def get_info():
    global status
    return status


# asynchronously start client
@app.post("/start")
async def client_start_trigger(background_tasks: BackgroundTasks, manager_data: client_utils.ManagerData):
    global status

    # client_manager address
    client_res = client_api.ClientMangerAPI().get_info()

    # latest global model version
    latest_gl_model_v = client_res.json()['GL_Model_V']

    # next global model version
    status.FL_next_gl_model = latest_gl_model_v + 1

    logging.info('bulid model')

    logging.info('FL start')
    status.FL_client_start = True

    status.FL_server_IP = manager_data.server_ip
    status.FL_task_id = task_id
    status.FL_client_mac = client_utils.get_mac_address()

    # get the FL server IP
    status.FL_server_IP = client_api.ClientServerAPI(status.FL_task_id).get_port()

    # start FL Client
    background_tasks.add_task(fl_client_start)

    return status


async def fl_client_start():
    logging.info('FL learning ready')
    global status

    """
    Before running this code, 
    set wandb api and account in the config.yaml
    """
    # Set the name in the wandb project
    wandb_name = f"client-v{status.FL_next_gl_model}({datetime.now()})"
    # Login and init wandb project
    wandb_run = client_wandb.start_wandb(config, wandb_name)

    # Register client task
    x_train, x_test, y_train, y_test, y_label_counter, model, model_name = client_task.register_task(task_id, dataset, label_count)

    try:
        # Update wandb config
        wandb_config = {"learning_rate": model.optimizer.learning_rate.numpy(), "optimizer": model.optimizer._name,
                        "dataset": dataset, "model_architecture": model_name}
        wandb_run.config.update(wandb_config, allow_val_change=True)

        # Check data status in the wandb
        label_values = [[i, y_label_counter[i]] for i in range(label_count)]
        client_wandb.data_status_wandb(wandb_run, label_values)

        loop = asyncio.get_event_loop()
        client = client_fl.FLClient(model, x_train, y_train, x_test, y_test, validation_split, status.FL_task_id, status.FL_client_mac,
                                    fl_round=1, next_gl_model=status.FL_next_gl_model, wandb_name=wandb_name, wandb_run=wandb_run, model_name=model_name)

        # client_start object
        client_start = client_fl.flower_client_start(status.FL_server_IP, client)

        # FL client start time
        fl_start_time = time.time()

        # Run asynchronously FL client
        await loop.run_in_executor(None, client_start)

        logging.info('fl learning finished')

        # FL client end time
        fl_end_time = time.time() - fl_start_time

        # Wandb log(Client round end time)
        wandb_run.log({"operation_time": fl_end_time, "next_gl_model_v": status.FL_next_gl_model},step= status.FL_next_gl_model)

        client_all_time_result = {"fl_task_id": status.FL_task_id, "client_mac": status.FL_client_mac, "operation_time": fl_end_time,
                                  "next_gl_model_v": status.FL_next_gl_model, "wandb_name": wandb_name}
        json_result = json.dumps(client_all_time_result)
        logging.info(f'client_operation_time - {json_result}')

        # Send client_time_result to client_performance pod
        client_api.ClientServerAPI(status.FL_task_id).put_client_time_result(json_result)

        # Get client system result from wandb and send it to client_performance pod
        client_wandb.client_system_wandb(task_id, status.FL_client_mac, status.FL_next_gl_model, wandb_name)

        # Close wandb
        client_wandb.client_finish_wandb()

        # Delete client object
        del client

        # Complete Client training
        status.FL_client_start = await client_utils.notify_fin()
        logging.info('FL Client Learning Finish')

    except Exception as e:
        logging.info('[E][PC0002] learning', e)
        status.FL_client_fail = True
        status.FL_client_start = await client_utils.notify_fail()
        raise e


def main():
    try:
        # create client api => to connect client manager
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)

    finally:
        # FL client out
        client_api.ClientMangerAPI().get_client_out()
        logging.info(f'{status.FL_client_mac} client close')
