import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import logging
import uuid, socket
from . import client_api
from flwr.common.typing import NDArrays
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import torch
import json
from collections import OrderedDict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# set log format
handlers_list = [logging.StreamHandler()]

if "MONITORING" in os.environ:
    if os.environ["MONITORING"] == '1':
        handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
    else:
        pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


# FL Client Status class
class FLClientStatus(BaseModel):
    task_id: str = ''
    client_num: int = 0
    client_mac: str = ''
    client_name: str = ''
    client_online: bool = True
    client_start: bool = False
    client_fail: bool = False
    server_IP: str = None # FL server IP
    gl_model: int = 0


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i + 2] for i in range(0, 12, 2)])


def get_hostname():
    return socket.gethostname()

# make local model directory
def local_model_directory(task_id):
    local_list = []
    # Local Model repository
    if not os.path.isdir(f'./local_model'):
        os.mkdir(f'./local_model')
        
    if not os.path.isdir(f'./local_model/{task_id}'):
        os.mkdir(f'./local_model/{task_id}')
        local_list = os.listdir(f'./local_model/{task_id}')
        
    else:
        local_list = os.listdir(f'./local_model/{task_id}')

    return local_list


# latest local model download
def download_local_model(model_type, task_id, listdir, model=None):

    pattern = r"([A-Za-z]+)_local_model_V(\d+)\.(h5|pth)"
    matching_files = [f for f in listdir if re.match(pattern, f)]

    if matching_files:
        last_local_model_file = sorted(matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
        # model_name = re.findall(pattern, latest_local_model_file)[0][0]
        model_extension = re.findall(pattern, last_local_model_file)[0][2]
        model_path = os.path.join(f"./local_model/{task_id}/", last_local_model_file)

        logging.info(f'downloaded local_model_name: {last_local_model_file}')
        
        # Initialize downloaded model
        # downloaded_model = None

        if model_type == "Tensorflow" and model_extension == "h5":
            # Load TensorFlow Keras model
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)

        elif model_type == "Pytorch" and model_extension == "pth":
            import torch
            model.load_state_dict(torch.load(model_path))

    else:
        print("No matching model files found.")

    return model


# check train finish info to client manager
async def notify_fin():
    logging.info('try notify_fin')
    FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, client_api.ClientMangerAPI().get_train_fin())
    r = await future2
    
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.error(f'notify_fin error: {r.content}')

    return FL_client_start


# check train fail info to client manager
async def notify_fail():

    logging.info('notify_fail start')

    FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, client_api.ClientMangerAPI().get_train_fail())
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return FL_client_start


""" For LLM Functions """
def set_parameters_for_llm(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def get_parameters_for_llm(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

def load_model(model_name: str, quantization: int, gradient_checkpointing: bool, peft_config):
    if quantization == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        bnb_config = None  # 양자화 안함

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    if gradient_checkpointing:
        model.config.use_cache = False

    return get_peft_model(model, peft_config)

def gen_parameter_shape(cfg) -> None:
    """
    Huggingface 모델에 LoRA를 적용한 후, 학습 가능한 파라미터들의 shape을 JSON으로 저장
    cfg: OmegaConf or dict, 반드시 cfg.model.name이 포함되어 있어야 함
    """
    model_name = cfg.model.name
    save_filename = cfg.save_filename if "save_filename" in cfg else "parameter_shapes.json"

    # 모델 로드 및 LoRA 적용
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = LoraConfig(
        r=cfg.finetune.lora_r,
        lora_alpha=cfg.finetune.lora_alpha,
        lora_dropout=cfg.finetune.lora_dropout,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)

    # LoRA 파라미터 shape 추출
    parameter_shapes = [list(p.shape) for p in get_parameters_for_llm(peft_model)]

    # 현재 실행 위치 기준으로 저장
    save_path = os.path.abspath(save_filename)

    # JSON 저장
    with open(save_path, "w") as f:
        json.dump(parameter_shapes, f, indent=2)

    print(f"[✓] {save_filename} saved to: {save_path}")