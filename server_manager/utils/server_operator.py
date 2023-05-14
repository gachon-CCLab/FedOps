from kubernetes import client, config, watch
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
import time


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


def get_running_tasks(namespace: str = 'fedops'):
    load_config()

    # Initialize the Kubernetes client
    api_instance = client.CoreV1Api()

    # List all pods in the namespace
    pods = api_instance.list_namespaced_pod(namespace)

    running_tasks = []

    for pod in pods.items:
        # Check if the pod is a 'fl-server-job-' pod and if it is running
        # if pod.metadata.name.startswith('fl-server-job-') and pod.status.phase == 'Running':
        if pod.metadata.name.startswith('fl-server-job-'):
            # The task_id is the third part of the pod name when split by '-'
            task_id = pod.metadata.name.split('-')[3]
            running_tasks.append(task_id)

    return running_tasks


def get_unused_port(namespace: str = 'fedops'):
    load_config()

    running_tasks = get_running_tasks(namespace)

    api_instance = client.CustomObjectsApi()

    virtual_service_name = 'fedops-virtualservice'

    try:
        # Try to get the existing VirtualService
        virtual_service = api_instance.get_namespaced_custom_object(
            group="networking.istio.io",
            version="v1alpha3",
            namespace=namespace,
            plural="virtualservices",
            name=virtual_service_name
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # If the VirtualService does not exist, no ports are being used
            return 40021
        else:
            raise e

    used_ports = []
    new_routes = []

    print("virtual_service['spec']['tcp']:", virtual_service["spec"]["tcp"])

    if isinstance(virtual_service["spec"]["tcp"], list):
        for route in virtual_service["spec"]["tcp"]:
            port = route["match"][0]["port"]
            # Avoid removing the example route
            if port == 40000:
                new_routes.append(route)
                continue
            task_id = route["route"][0]["destination"]["host"].split('-')[3]
            # If the task for this port is running, add it to the new list
            if task_id in running_tasks:
                new_routes.append(route)
                used_ports.append(port)
            print("route - ", route)
    else:
        print("virtual_service['spec']['tcp'] is not a list")

    # Replace the "tcp" field with the new routes
    virtual_service["spec"]["tcp"] = new_routes

    # Update the VirtualService to remove the routes for the non-running tasks
    if virtual_service["spec"]["tcp"]:
        try:
            api_instance.patch_namespaced_custom_object(
                group="networking.istio.io",
                version="v1alpha3",
                namespace=namespace,
                plural="virtualservices",
                name=virtual_service_name,
                body=virtual_service,
            )
            print("Updated Istio VirtualService")
        except client.exceptions.ApiException as e:
            raise e

    # Return the first unused port in the range
    for port in range(40021, 40041):
        if port not in used_ports:
            return port

    raise Exception("No unused ports available")


def update_virtual_service(task_id: str, service_name: str, port: int, namespace: str):
    load_config()

    api_instance = client.CustomObjectsApi()

    virtual_service_name = 'fedops-virtualservice'
    try:
        # Try to get the existing VirtualService
        virtual_service = api_instance.get_namespaced_custom_object(
            group="networking.istio.io",
            version="v1alpha3",
            namespace=namespace,
            plural="virtualservices",
            name=virtual_service_name
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # If the VirtualService does not exist, create a new one
            virtual_service = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": virtual_service_name,
                    "namespace": namespace
                },
                "spec": {
                    "hosts": ["fedops.svc.cluster.local"],
                    "tcp": []
                }
            }
        else:
            raise e

    # Add the new route to the VirtualService
    new_route = {
        "match": [{
            "port": port
        }],
        "route": [
            {
                "destination": {
                    "host": f"{service_name}.{namespace}.svc.cluster.local",
                    "port": {"number": 80}
                }
            }
        ]
    }
    if isinstance(virtual_service["spec"]["tcp"], list):
        virtual_service["spec"]["tcp"].append(new_route)
    else:
        virtual_service["spec"]["tcp"] = [new_route]

    # Update or create the VirtualService
    try:
        api_instance.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1alpha3",
            namespace=namespace,
            plural="virtualservices",
            name=virtual_service_name,
            body=virtual_service,
        )
        print(f"Updated Istio VirtualService for task_id: {task_id}")
    except client.exceptions.ApiException as e:
        if e.status == 404:
            try:
                api_instance.create_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1alpha3",
                    namespace=namespace,
                    plural="virtualservices",
                    body=virtual_service,
                )
                print(f"Created Istio VirtualService for task_id: {task_id}")
            except client.exceptions.ApiException as e_create:
                raise e_create
        else:
            raise e



def create_fl_server(task_id: str, fl_server_status: dict):
    load_config()

    fl_server_status[task_id]["status"] = "Creating"



    job_name = "fl-server-job-" + task_id
    pod_name_prefix = "fl-server-"

    # Initialize the Kubernetes client for batch jobs
    api_instance = client.BatchV1Api()

    # Initialize the Kubernetes client for custom objects
    custom_api_instance = client.CustomObjectsApi()

    # Check if a job with the same name already exists
    namespace = "fedops"
    existing_jobs = api_instance.list_namespaced_job(
        namespace,
        field_selector=f"metadata.name={job_name}"
    )

    if len(existing_jobs.items) > 0:
        # A job with the same name exists
        existing_job = existing_jobs.items[0]
        job_status = existing_job.status.conditions[-1].type if existing_job.status.conditions else None
        if job_status in ['Complete', 'Failed']:
            # If the job is complete or failed, delete it before creating a new one
            print(f"Deleting existing job with name {job_name} because its status is {job_status}.")
            api_instance.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy='Foreground')
            )
            # Wait for the job to be deleted
            w = watch.Watch()
            for event in w.stream(api_instance.list_namespaced_job, namespace=namespace):
                if event['object'].metadata.name == job_name and event['type'] == 'DELETED':
                    print(f"Job {job_name} deleted.")
                    w.stop()
                    break
        else:
            print(f"Job with name {job_name} already exists and is not Complete or Failed. Skipping job creation.")
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
            backoff_limit=0
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

    port = get_unused_port()
    fl_server_status[task_id]["port"] = port

    update_virtual_service(task_id, service_name, port, namespace)

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
                # When the job has completed or failed, delete the job
                if current_job.status.succeeded == 1 or current_job.status.failed:
                    print("Deleting job")
                    api_instance.delete_namespaced_job(
                        name=job_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions(propagation_policy='Foreground')
                    )

                    # Delete the corresponding service
                    print("Deleting service")
                    core_v1_api.delete_namespaced_service(
                        name=service_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions()
                    )

                    # Remove the route from the VirtualService
                    print("Removing route from VirtualService")
                    virtual_service = custom_api_instance.get_namespaced_custom_object(
                        group="networking.istio.io",
                        version="v1alpha3",
                        namespace=namespace,
                        plural="virtualservices",
                        name="fedops-virtualservice"
                    )
                    new_routes = [route for route in virtual_service["spec"]["tcp"]
                                  if route["route"][0]["destination"][
                                      "host"] != f"{service_name}.{namespace}.svc.cluster.local"]
                    virtual_service["spec"]["tcp"] = new_routes
                    custom_api_instance.patch_namespaced_custom_object(
                        group="networking.istio.io",
                        version="v1alpha3",
                        namespace=namespace,
                        plural="virtualservices",
                        name="fedops-virtualservice",
                        body=virtual_service,
                    )
                    break
    except Exception as e:
        print(f"Error while monitoring job status: {e}")
        fl_server_status[task_id]["status"] = "Error"
    finally:
        print("Cleaning up resources")
        w.stop()