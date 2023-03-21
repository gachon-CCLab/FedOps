from datetime import datetime, timedelta

from kubernetes.client import models as k8s
from airflow.models import DAG, Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.kubernetes.secret import Secret
from airflow.kubernetes.pod import Resources
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)

dag_id = 'fl-ios-server-dag'

now = datetime.now()

task_default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(hours=6),
    'start_date': datetime(now.year, now.month, now.day),
    'depends_on_past': False,
    'email': ['tpah20@gmail.com'],
    'email_on_retry': False,
    'email_on_failure': False,
    'execution_timeout': timedelta(hours=4)
}

dag = DAG(
    dag_id=dag_id,
    description='fl android server pod operator',
    default_args=task_default_args,
    schedule_interval='5 16 * * *',
    max_active_runs=1
)

env = Secret(
    'env',
    'TEST',
    'test_env',
    'TEST',
)
env1 = Secret(
    deploy_type='env',
    deploy_target='ACCESS_KEY_ID',
    secret='s3secret',
    key='ACCESS_KEY_ID',
)
env2 = Secret(
    deploy_type='env',
    deploy_target='ACCESS_SECRET_KEY',
    secret='s3secret',
    key='ACCESS_SECRET_KEY',
)
env3 = Secret(
    deploy_type='env',
    deploy_target='BUCKET_NAME',
    secret='s3secret',
    key='BUCKET_NAME',
)
# env4 = Secret(
#     deploy_type='env',
#     deploy_target='WB_KEY',
#     secret='wandb',
#     key='WB_KEY',
# )
pod_resources = Resources()
pod_resources.request_cpu = '2000m'
pod_resources.request_memory = '2048Mi'
# pod_resources.limit_cpu = '2000m'
# pod_resources.limit_memory = '4096Mi'
# pod_resources.limit_gpu = '1'


configmaps = [
    k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name='secret')),
]

start = DummyOperator(task_id="start", dag=dag)
port = k8s.V1ContainerPort(container_port=8080)

run = KubernetesPodOperator(
    task_id="kubernetespodoperator",
    namespace='fedmlops',
    image='docker.io/hoo0681/airflowkubepodimage:0.1',
    #cmds=["python3"],
    #cmds=["/bin/sh","-c","apt-get install -y curl; until curl -fsl http://localhost:4191/ready; \
    #    do echo \"Waiting for Sidecar...\"; sleep 3; done; echo \"Sidecar available. Running the command...\"; \
    #    git clone -b ${GIT_TAG} ${REPO_URL} /app; \
    #    python3 -m pip install -r /app/requirements.txt; \
    #    python3 /app/app.py; \
    #    x=$(echo $?); curl -fsI -X POST http://localhost:4191/shutdown && exit $x"],
    #arguments=["/app/app.py"],
    cmds=["/bin/sh","-c","git clone -b ${GIT_TAG} ${REPO_URL} /app; \
        python3 -m pip install -r /app/requirements.txt; \
        python3 /app/app.py;"],
    ports=[port],
    labels={'run':'fl-ios-server'},
    env_vars={'REPO_URL':'https://github.com/gachon-CCLab/flower_ios_server.git',
              "GIT_TAG":"master" ,
              "ENV": 'init' },
    #secrets=[
    #    env
    #],
    #image_pull_secrets=[k8s.V1LocalObjectReference('image_credential')],
    name="fl-ios-server",
    is_delete_operator_pod=True,
    get_logs=True,
    resources=pod_resources,
    secrets=[env1,env2,env3],
    #env_from=configmaps,
    dag=dag,
)
##################################

start >> run
