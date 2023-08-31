import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import logging
import yaml, uuid, socket
from . import client_api

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


# FL Client Status class
class FLClientStatus(BaseModel):
    FL_task_id: str = ''
    FL_client_num: int = 0
    FL_client_mac: str = ''
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_next_gl_model: int = 0


# Get manager data info
class ManagerData(BaseModel):
    server_ip: str
    client_mac: str


def read_config(file_path):
    # Read the YAML configuration file
    config_file_path = file_path
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i + 2] for i in range(0, 12, 2)])


def get_hostname():
    return socket.gethostname()

# get torch model parameters
def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# torch train
def torch_train(model, train_dataset, criterion, optimizer, validation_split, epochs, batch_size, device: str = "cpu"):
    from torch import nn
    from torch import optim
    from torch.utils.data import DataLoader, Subset
    """Train the network on the training set."""
    print("Starting training...")

    model.to(device)  # move model to GPU if available
    
    n_valset = int(len(train_dataset) * validation_split)

    valset = Subset(train_dataset, range(0, n_valset))
    trainset = Subset(
        train_dataset, range(n_valset, len(train_dataset))
    )

    # DataLoader for training and validation
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            inputs, labels = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.to("cpu")  # move model back to CPU

    train_loss, train_acc = torch_test(model, train_loader, criterion=criterion)
    val_loss, val_acc = torch_test(model, val_loader, criterion=criterion)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


# torch test
def torch_test(model, test_loader, criterion, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    import torch
    print("Starting evalutation...")
    model.to(device)  # move model to GPU if available
    correct, loss = 0, 0.0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            labels_int = torch.argmax(labels, dim=1)
            loss += criterion(outputs, labels).item()
            predicted = torch.argmax(outputs, dim=1)
            # _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels_int).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(test_loader.dataset)
    model.to("cpu")  # move model back to CPU
    return loss, accuracy


# Check tensorflow or torch model
def identify_model(model):
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return "Tensorflow"
    except ImportError:
        pass
    
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "Pytorch"
    except ImportError:
        pass

    return "Unknown Model"


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
        latest_local_model_file = sorted(matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
        # model_name = re.findall(pattern, latest_local_model_file)[0][0]
        model_extension = re.findall(pattern, latest_local_model_file)[0][2]
        model_path = os.path.join(f"./local_model/{task_id}/", latest_local_model_file)

        logging.info(f'downloaded local_model_name: {latest_local_model_file}')
        
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