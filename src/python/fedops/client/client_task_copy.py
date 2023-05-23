import client_data
import client_model
import client_utils
import client_wandb
import app
import logging, os

# set log format
handlers_list = [logging.StreamHandler()]

if os.environ["MONITORING"] == '1':
    handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
else:
    pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

def register_task(task_id, dataset, label_count):
    # Set client number
    FL_client_num = 0

    """
   Client data load function
   Split partition => apply each client dataset(Options)
   After setting data method in client_data.py, call the data method.
   Keep these variables.
   """
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