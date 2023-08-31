import argparse
import fl_data
import fl_model

import sys
sys.path.append('/home/ccl/Desktop/FedOps/src/python')

from fedops.client import client_utils
from fedops.client import app
import logging

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

def register_task(FL_client_num):

    """
   Client data load function
   Split partition => apply each client dataset(Options)
   After setting data method in client_data.py, call the data method.
   Keep these variables.
   """
    train_loader, val_loader, test_loader, y_label_counter = fl_data.load_partition(dataset=config['client']['data']['name'], 
                                                                        FL_client_num=FL_client_num, 
                                                                        validation_split=config['client']['data']['validation_split'], 
                                                                        label_count=config['client']['data']['label_count'],
                                                                        batch_size=config['server']['fl_round']['batch_size']) # Pytorch version
    # (x_train, y_train), (x_test, y_test), y_label_counter = client_data.load_partition(dataset, FL_client_num, label_count) # Tensorflow version

    logger.info('data loaded')

    """
    #     Client local model build function
    #     Set init local model
    #     After setting model method in client_model.py, call the model method.
    #     """
    # torch model
    model = fl_model.DNN(input_dim=6, num_classes=config['client']['data']['label_count'])
    criterion, optimizer = fl_model.set_model_parameter(model)
    model_name = model.model_name
    train_torch = fl_model.train_torch() # set torch train
    test_torch = fl_model.test_torch() # set torch test

    # Check tensorflow or torch model
    model_type = client_utils.identify_model(model)

    # Local model directory for saving local models
    task_id = config['client']['task']['name']  # FL task ID
    local_list = client_utils.local_model_directory(task_id)

    # If you have local model, download latest local model 
    if local_list:
        logger.info('Latest Local Model download')
        # If you use torch model, you should input model variable in model parameter
        model = client_utils.download_local_model(model_type=model_type, task_id=task_id, listdir=local_list, model=model)  
        
    registration = {
        "model_type" : model_type,
        "train_loader" : train_loader,
        "val_loader" : val_loader,
        "test_loader" : test_loader,
        "y_label_counter" : y_label_counter,
        "criterion" : criterion,
        "optimizer" : optimizer,
        "model" : model,
        "model_name" : model_name,
        "train_torch" : train_torch,
        "test_torch" : test_torch
    } # torch version
    
    # tesorflow version
    # registration = {"model_type": model_type, "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test":y_test, "y_label_counter": y_label_counter, "model": model, "model_name":model_name}
    
    
    return registration
    # return model_type, train_loader, val_loader, test_loader, y_label_counter, criterion, optimizer, model, model_name, train_torch, test_torch
    # return model_type, x_train, x_test, y_train, y_test, y_label_counter, model, model_name # Tensorflow version



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a number with different operations.')
    parser.add_argument('--number', '-n', type=int, required=True, help='An integer number.')
    parser.add_argument('--port', '-p', type=int, required=True, help='An integer number.')
    
    args = parser.parse_args()
    
    FL_client_num = args.number
    FL_client_port = args.port
    # FL_client_num = 2
    
    # read config.yaml file
    config_file_path = '/home/ccl/Desktop/FedOps/examples/config.yaml'
    config = client_utils.read_config(config_file_path)
    
    fl_task = register_task(FL_client_num)
    fl_client = app.FLClientTask(config, fl_task, FL_client_port)
    fl_client.start()

