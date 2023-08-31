import client_data
import client_model

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

# read config.yaml file
config_file_path = '../config.yaml'
config = client_utils.read_config(config_file_path)

# FL task ID
task_id = config['task']['name']

# Data name
dataset = config['data']['name']

# Label count
label_count = config['data']['label_count']

# Train validation split
validation_split = config['data']['validation_split']

def register_task():
    # Set client number
    FL_client_num = 0

    """
   Client data load function
   Split partition => apply each client dataset(Options)
   After setting data method in client_data.py, call the data method.
   Keep these variables.
   """
    # x_train, y_train, x_test, y_test, y_label_counter = 1,2,3,4,5
    (x_train, y_train), (x_test, y_test), y_label_counter = client_data.load_partition(dataset, FL_client_num, label_count)
    logger.info('data loaded')

    # Local model directory for saving local models
    local_list = client_utils.local_model_directory(task_id)

    if not local_list:
        # Build init local model
        logging.info('init local model')
        """
        Client local model build function
        Set init local model
        After setting model method in client_model.py, call the model method.
        """
        model, model_name = client_model.CNN()

    else:
        # Download latest local model
        logger.info('Latest Local Model download')
        model, model_name = client_utils.download_local_model(task_id, local_list)

    return x_train, x_test, y_train, y_test, y_label_counter, model, model_name


if __name__ == "__main__":
    fl_task = register_task()
    fl_client = app.FLClientTask(config, fl_task)
    fl_client.start()

