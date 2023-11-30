import logging
import wandb
from . import client_api


def start_wandb(wandb_key, wandb_project, wandb_name):
    wandb.login(key=wandb_key)
    config_wandb = {
        "learning_rate": 0,
        "optimizer": '',
        "dataset": '',
        "model_architecture": '',
        "batch_size": 0,
        "epochs": 0,
        "num_rounds": 0
    }
    run = wandb.init(project=wandb_project, name=wandb_name, config=config_wandb)

    return run


def data_status_wandb(run=None, labels=None):
    table = wandb.Table(data=labels, columns=["label", "data_size"])
    run.log({'Data Lable Histogram': wandb.plot.bar(table, "label", "data_size", title="Data Size Distribution")})


def client_system_wandb(fl_task_id, client_mac, client_name, next_gl_model_v, wandb_name, wandb_account):
    try:
        # check client system resource usage from wandb
        api = wandb.Api()
        runs = api.runs(f"{wandb_account}/{fl_task_id}")

        sys_df = runs[0].history(stream="system")

        cols = ['system.network.sent', 'system.network.recv', 'system.disk', '_runtime', 'system.proc.memory.rssMB','system.proc.memory.availableMB', 'system.cpu', 'system.proc.cpu.threads', 'system.memory', 'system.proc.memory.percent', '_timestamp']

        sys_df = sys_df[cols]

        sys_df.rename(columns={
            "system.network.sent": "network_sent",
            "system.network.recv": "network_recv",
            "system.disk": "disk",
            "_runtime": "runtime",
            "system.proc.memory.rssMB": "memory_rssMB",
            "system.proc.memory.availableMB": "memory_availableMB",
            "system.cpu": "cpu",
            "system.proc.cpu.threads": "cpu_threads",
            "system.memory": "memory",
            "system.proc.memory.percent": "memory_percent",
            "_timestamp": "timestamp",
        }, inplace=True)

        # Extract df_row by row
        for i in range(len(sys_df)):
            sys_df_row = sys_df.iloc[i].copy()
            sys_df_row['fl_task_id'] = fl_task_id
            sys_df_row['client_mac'] = client_mac
            sys_df_row['client_name'] = client_name
            sys_df_row['next_gl_model_v'] = next_gl_model_v
            sys_df_row['wandb_name'] = wandb_name

            sys_df_row_json = sys_df_row.to_json()

            # send client_system  to client_performance pod
            client_api.ClientServerAPI(fl_task_id).put_client_system(sys_df_row_json)

    except Exception as e:
        logging.error(f'wandb system load error: {e}')

