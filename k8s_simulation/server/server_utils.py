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


def read_config():
    # Read the YAML configuration file
    config_file_path = './config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



# 참고: https://loosie.tistory.com/210, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
# aws session 연결
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                 aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
                                 region_name=region_name)


# s3에 global model upload
def upload_model_to_bucket(task_id, global_model, next_gl_model_v, model_name):
    bucket_name = os.environ.get('BUCKET_NAME')

    logging.info(f'{model_name}_gl_model_%{next_gl_model_v}_V.h5 모델 업로드 시작')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/app/{model_name}_gl_model_V{next_gl_model_v}.h5',
        Key=f'{task_id}/{global_model}',
    )

    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model}"
    logging.info(f'Upload {model_name}_gl_model_v{next_gl_model_v}.h5')
    return s3_url


# s3에 저장되어 있는 latest global model download
def model_download(task_id):
    bucket_name = os.environ.get('BUCKET_NAME')
    # print('bucket_name: ', bucket_name)

    try:
        session = aws_session()
        s3_resource = session.client('s3')
        bucket_list = s3_resource.list_objects(Bucket=bucket_name)
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

        return gl_model, gl_model_version

    # s3에 global model 없을 경우
    except Exception as e:
        logging.error('No read global model')
        model_X = None
        gl_model_version=0
        logging.info(f'gl_model: {model_X}, gl_model_v: {gl_model_version}')

        return model_X, gl_model_version