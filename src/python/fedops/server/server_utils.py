import yaml
import boto3
import os, logging, re
from abc import ABC, abstractmethod


# FL Server Status Class
class FLServerStatus:
    latest_gl_model_v = 0  # Previous Global Model Version
    next_gl_model_v = 0  # Global model version to be created
    start_by_round = 0  # fit aggregation start
    end_by_round = 0  # fit aggregation end
    round = 0  # round number


class Torch(ABC):
    @abstractmethod
    def torch_test(self, model, val_loader, criterion, steps: int = None, device: str = "cpu"):
        pass
    

def read_config(file_path):
    # Read the YAML configuration file
    config_file_path = file_path
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

# Connect aws session
# def aws_session(region_name='ap-northeast-2'):
#     return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
#                                  aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
#                                  region_name=region_name)
def aws_session(region_name='ap-northeast-2'):
    ACCESS_KEY_ID = "AKIA2ENEMY666LKZE3W4"
    ACCESS_SECRET_KEY = "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"
    return boto3.session.Session(aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=ACCESS_SECRET_KEY,
                                 region_name=region_name)


# Global model upload in S3
def upload_model_to_bucket(task_id, global_model_name):
    # bucket_name = os.environ.get('BUCKET_NAME')
    bucket_name = "global-model"

    logging.info(f'bucket_name: {bucket_name}')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'./{global_model_name}',
        Key=f'{task_id}/{global_model_name}',
    )

    logging.info(f'Upload {global_model_name}')

    # s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model_name}"
    # return s3_url



# torch test
def torch_test(model, val_loader, criterion, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    import torch
    print("Starting evalutation...")
    model.to(device)  # move model to GPU if available
    correct, loss = 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            predicted = torch.argmax(outputs, dim=1)
            labels_int = torch.argmax(labels, dim=1)
            # logging.info(f"predicted: {predicted}")
            # logging.info(f"labels_int: {labels_int}")
            # _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels_int).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(val_loader.dataset)
    model.to("cpu")  # move model back to CPU
    return loss, accuracy

# Download the latest global model stored in s3
def model_download_s3(task_id, model_type, model=None):
    # bucket_name = os.environ.get('BUCKET_NAME')
    bucket_name = "global-model"
    # print('bucket_name: ', bucket_name)

    try:
        session = aws_session()
        s3_resource = session.client('s3')
        bucket_list = s3_resource.list_objects_v2(Bucket=bucket_name, Prefix=f'{task_id}/')
        content_list = bucket_list['Contents']

        # Inquiry global model file in s3 bucket
        file_list = []

        for content in content_list:
            key = content['Key']
            file_name = key.split('/')[1]
            file_list.append(file_name)

        logging.info(f'model_file_list: {file_list}')
        
        # File name pattern
        pattern = r"([A-Za-z]+)_gl_model_V(\d+)\.(h5|pth)"

        if file_list:
            latest_gl_model_file = sorted(file_list, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
            gl_model_name = re.findall(pattern, latest_gl_model_file)[0][0]
            gl_model_version = int(latest_gl_model_file.split('_V')[1].split('.')[0])
            gl_model_path = os.path.join(f"{task_id}/", latest_gl_model_file)

        gl_model_save_path = f'./{latest_gl_model_file}'
        s3_resource.download_file(bucket_name, gl_model_path, gl_model_save_path)
        
        
        if model_type == "Tensorflow":
            # Load TensorFlow Keras model
            import tensorflow as tf
            model = tf.keras.models.load_model(gl_model_save_path)

        elif model_type == "Pytorch":
            import torch
            model.load_state_dict(torch.load(gl_model_save_path))

        # gl_model = tf.keras.models.load_model(gl_model_save_path)            

        return model, gl_model_name, gl_model_version
        # return gl_model, gl_model_name, gl_model_version


    except Exception as e:
        logging.error('No read global model')
        gl_model = None
        gl_model_name = None
        gl_model_version=0
        logging.info(f'gl_model: {gl_model}, gl_model_v: {gl_model_version}')

        return gl_model, gl_model_name, gl_model_version
    
    
def model_download_local(model_type, model=None):
    
    local_list = os.listdir(f"./")
        
    # File name pattern
    pattern = r"([A-Za-z]+)_gl_model_V(\d+)\.(h5|pth)"
        
    if local_list:
        matching_files = [x for x in local_list if re.match(pattern, x)]
        
        if matching_files:
            latest_gl_model_file = sorted(matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
            gl_model_name = re.findall(pattern, latest_gl_model_file)[0][0]
            gl_model_version = int(re.findall(pattern, latest_gl_model_file)[0][1])
            
            if model_type == "Tensorflow":
                # Load TensorFlow Keras model
                import tensorflow as tf
                model = tf.keras.models.load_model(latest_gl_model_file)
                
            elif model_type == "Pytorch":
                import torch
                logging.info(f"loaded model parameters")
                model.load_state_dict(torch.load(latest_gl_model_file))
                
            else:
                print("No matching model files found.")
                return None, None, 0

            return model, gl_model_name, gl_model_version
        
        else:
            print("No matching model files found.")
            return None, None, 0

    else: 
        print('No read global model')
        gl_model = None
        gl_model_name = None
        gl_model_version = 0

        return gl_model, gl_model_name, gl_model_version
    