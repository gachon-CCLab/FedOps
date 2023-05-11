from kubernetes import client, config, watch
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
import time
import random

# Keep track of assigned ports
assigned_ports = None


def find_available_port(fl_server_status: dict, task_id: str, namespace: str):
    ingress_name = "fl-server-ingress"
    min_port: int = 40021
    max_port: int = 40040

    # Load the existing Ingress resource
    api_instance = client.NetworkingV1Api()
    ingress = api_instance.read_namespaced_ingress(ingress_name, namespace)

    used_ports = set()
    for rule in ingress.spec.rules:
        if "ccljhub.gachon.ac.kr" in rule.host:
            used_port = int(rule.host.split(':')[1])
            used_ports.add(used_port)

    for port in range(min_port, max_port + 1):
        if port not in used_ports:
            fl_server_status[task_id]["port"] = port
            return port

    raise ValueError("No available port found within the specified range")



def release_assigned_port(port: int):
    global assigned_ports
    assigned_ports.discard(port)


def load_config():
    try:
        # Try loading the in-cluster configuration
        config.load_incluster_config()
        print("Using in-cluster config")
    except config.ConfigException:
        # Fallback to loading the kubeconfig file
        config.load_kube_config('config.txt')
        print("Using kubeconfig file")

    return config


# Add this new variable at the beginning of the script
base_path = "/fedops/server/fl-server"


def update_ingress_with_service(task_id: str, service_name: str, namespace: str, assigned_port: int):
    load_config()
    ingress_name = "fl-server-ingress"

    # Load the existing Ingress resource
    api_instance = client.NetworkingV1Api()
    ingress = api_instance.read_namespaced_ingress(ingress_name, namespace)

    # Modify the host to include the assigned port
    host = f"ccljhub.gachon.ac.kr:{assigned_port}"

    new_rule = client.V1IngressRule(
        host=host,
        http=client.V1HTTPIngressRuleValue(
            paths=[
                client.V1HTTPIngressPath(
                    path="/",
                    path_type="Prefix",
                    backend=client.V1IngressBackend(
                        service=client.V1IngressServiceBackend(
                            name=service_name,
                            port=client.V1ServiceBackendPort(number=80)
                        )
                    )
                )
            ]
        )
    )

    # Check if a rule with the same host already exists
    existing_rule = None
    for rule in ingress.spec.rules:
        if rule.host == new_rule.host:
            existing_rule = rule
            break

    if existing_rule:
        print(f"Found existing rule for host: {host}. Updating it.")
        existing_rule.http = new_rule.http
    else:
        # Add a new rule with the given host
        ingress.spec.rules.append(new_rule)

    # Update the Ingress resource
    api_instance.replace_namespaced_ingress(ingress_name, namespace, ingress)
    print(f"Updated Ingress with task_id: {task_id}")


def create_fl_server(task_id: str, fl_server_status: dict):
    load_config()
    fl_server_status[task_id]["status"] = "Creating"
    fl_server_status[task_id]["port"]: int

    job_name = "fl-server-job-" + task_id
    pod_name_prefix = "fl-server-"

    # Check if a job with the same name already exists
    api_instance = client.BatchV1Api()
    namespace = "fedops"
    existing_jobs = api_instance.list_namespaced_job(
        namespace,
        field_selector=f"metadata.name={job_name}"
    )

    if len(existing_jobs.items) > 0:
        print(f"Job with name {job_name} already exists. Skipping job creation.")
        return

    env_vars = [V1EnvVar(name="REPO_URL", value='https://github.com/gachon-CCLab/FedOps-Training-Server.git'),
                V1EnvVar(name="GIT_TAG", value="main"),
                V1EnvVar(name="ENV", value="init"),
                V1EnvVar(name="TASK_ID", value=task_id),

                # Use existing Kubernetes secrets for environment variables
                V1EnvVar(name="ACCESS_KEY_ID",
                         value_from=V1EnvVarSource(
                             secret_key_ref=V1SecretKeySelector(name='s3secret', key='ACCESS_KEY_ID'))),
                V1EnvVar(name="ACCESS_SECRET_KEY",
                         value_from=V1EnvVarSource(
                             secret_key_ref=V1SecretKeySelector(name='s3secret', key='ACCESS_SECRET_KEY'))),
                V1EnvVar(name="BUCKET_NAME",
                         value_from=V1EnvVarSource(
                             secret_key_ref=V1SecretKeySelector(name='s3secret', key='BUCKET_NAME')))]

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name
        ),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    generate_name=pod_name_prefix,
                    labels={"run": "fl-server", "task_id": task_id}
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="fl-server",
                            image="docker.io/hoo0681/airflowkubepodimage:0.1",
                            ports=[
                                client.V1ContainerPort(container_port=8080)
                            ],
                            command=["/bin/sh", "-c"],
                            args=[
                                "git clone -b ${GIT_TAG} ${REPO_URL} /app; "
                                "python3 -m pip install -r /app/requirements.txt; "
                                "python3 /app/app.py;"
                            ],
                            env=env_vars,
                        )
                    ],
                    restart_policy="Never"
                )
            ),
            backoff_limit=5
        )
    )

    api_instance = client.BatchV1Api()
    namespace = "fedops"

    api_response = api_instance.create_namespaced_job(namespace, job)
    print("Job created. status='%s'" % str(api_response.status))

    # Update the status in the shared dictionary
    fl_server_status[task_id]["status"] = "Created"

    # Create a service for the job's pod
    service_name = "fl-server-service-" + task_id

    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=service_name),
        spec=client.V1ServiceSpec(
            selector={"run": "fl-server", "task_id": task_id},
            ports=[client.V1ServicePort(port=80, target_port=8080)],
            type="LoadBalancer",  # Change the service type to LoadBalancer
        ),
    )

    core_v1_api = client.CoreV1Api()

    # Check if the service already exists
    try:
        existing_service = core_v1_api.read_namespaced_service(namespace=namespace, name=service_name)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            existing_service = None
        else:
            raise e

    if existing_service:
        # Update the existing service
        service.metadata.resource_version = existing_service.metadata.resource_version
        core_v1_api.replace_namespaced_service(name=service_name, namespace=namespace, body=service)
        print(f"Updated service: {service_name}")
    else:
        # Create a new service
        core_v1_api.create_namespaced_service(namespace=namespace, body=service)
        print(f"Created service: {service_name}")

    # Wait until the load balancer has assigned an external IP to the service
    while True:
        service = core_v1_api.read_namespaced_service(namespace=namespace, name=service_name)
        if service.status.load_balancer.ingress:
            external_ip = service.status.load_balancer.ingress[0].ip
            print(f"External IP of the LoadBalancer: {external_ip}")
            break
        else:
            print("Waiting for external IP...")
            time.sleep(1)  # Wait for 1 seconds before checking again

    # Find an available port
    available_port = find_available_port(fl_server_status, task_id, 'fedops')

    # Update the Ingress resource to route traffic to the newly created service
    update_ingress_with_service(task_id, service_name, namespace, available_port)

    # Start watching for the job status
    w = watch.Watch()

    try:
        for event in w.stream(api_instance.list_namespaced_job, namespace=namespace):
            current_job = event['object']
            current_job_name = current_job.metadata.name

            if current_job_name == job_name:
                # Save the generated pod name in fl_server_status
                pod_name = current_job.spec.template.metadata.generate_name
                fl_server_status[task_id]["pod_name"] = pod_name

                if current_job.status.succeeded == 1:
                    print("Job succeeded")
                    fl_server_status[task_id]["status"] = "Finished"
                    w.stop()
                elif current_job.status.failed:
                    print("Job failed")
                    fl_server_status[task_id]["status"] = "Failed"
                    w.stop()
                elif current_job.status.active:
                    print("Job is running")
                    fl_server_status[task_id]["status"] = "Running"
                else:
                    print("Job status unknown")
                    fl_server_status[task_id]["status"] = "Unknown"

                # When the job has completed or failed, delete the job
                if current_job.status.succeeded == 1 or current_job.status.failed:
                    print("Deleting job")
                    api_instance.delete_namespaced_job(
                        name=job_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions()
                    )
                    break
    except Exception as e:
        print(f"Error while monitoring job status: {e}")
        fl_server_status[task_id]["status"] = "Error"
    finally:
        print("Cleaning up resources")
        w.stop()