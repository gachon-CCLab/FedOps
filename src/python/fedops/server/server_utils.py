import yaml
import boto3
import os, logging, re
import tensorflow as tf

# FL Server Status Class
class FLServerStatus:
    latest_gl_model_v = 0  # Previous Global Model Version
    next_gl_model_v = 0  # Global model version to be created
    start_by_round = 0  # fit aggregation start
    end_by_round = 0  # fit aggregation end
    round = 0  # round number


def read_config(file_path):
    # Read the YAML configuration file
    config_file_path = file_path
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Connect aws session
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                 aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
                                 region_name=region_name)


# Global model upload in S3
def upload_model_to_bucket(task_id, global_model_name):
    bucket_name = os.environ.get('BUCKET_NAME')
    # bucket_name = os.getenv('BUCKET_NAME')

    # logging.info(f'Upload {global_model_name}')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/app/{global_model_name}',
        Key=f'{task_id}/{global_model_name}',
    )

    logging.info(f'Upload {global_model_name}')

    # s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model_name}"
    # return s3_url


# Download the latest global model stored in s3
def model_download(task_id):
    bucket_name = os.environ.get('BUCKET_NAME')
    # bucket_name = os.getenv('BUCKET_NAME')
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
        pattern = r"([A-Za-z]+)_gl_model_V(\d+)\.h5"

        if file_list:
            latest_gl_model_file = sorted(file_list, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True)[0]
            gl_model_name = re.findall(pattern, latest_gl_model_file)[0][0]
            gl_model_version = int(latest_gl_model_file.split('_V')[1].split('.h5')[0])
            gl_model_path = os.path.join(f"{task_id}/", latest_gl_model_file)

        gl_model_save_path = f'/app/{latest_gl_model_file}'
        s3_resource.download_file(bucket_name, gl_model_path, gl_model_save_path)

        gl_model = tf.keras.models.load_model(gl_model_save_path)

        return gl_model, gl_model_name, gl_model_version


    except Exception as e:
        logging.error('No read global model')
        gl_model = None
        gl_model_name = None
        gl_model_version=0
        logging.info(f'gl_model: {gl_model}, gl_model_v: {gl_model_version}')

        return gl_model, gl_model_name, gl_model_version