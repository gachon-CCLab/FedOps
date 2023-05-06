from kubernetes import client, config, watch


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


def update_ingress_with_service(task_id: str, service_name: str, namespace: str):
    load_config()
    ingress_name = "fl-server-ingress"

    # Load the existing Ingress resource
    api_instance = client.NetworkingV1Api()
    ingress = api_instance.read_namespaced_ingress(ingress_name, namespace)

    # Add a new path with the given task_id
    new_path = client.V1HTTPIngressPath(
        path=f"/{task_id}(/|$)(.*)",
        path_type="Prefix",
        backend=client.V1IngressBackend(
            service=client.V1IngressServiceBackend(
                name=service_name,
                port=client.V1ServiceBackendPort(number=80)
            )
        )
    )

    ingress.spec.rules[0].http.paths.append(new_path)

    # Update the Ingress resource
    api_instance.replace_namespaced_ingress(ingress_name, namespace, ingress)
    print(f"Updated Ingress with task_id: {task_id}")


def create_fl_server(task_id: str, fl_server_status: dict):
    load_config()
    fl_server_status[task_id]["status"] = "Creating"

    job_name = "fl-server-job-" + task_id
    pod_name_prefix = "fl-server-"

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
                    labels={"app": "fl-server", "task_id": task_id}  # Add this line
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="fl-server",
                            image="your_image",
                            ports=[
                                client.V1ContainerPort(container_port=8080)
                            ]
                        )
                    ],
                    restart_policy="Never"
                )
            ),
            backoff_limit=1
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
            selector={"app": "fl-server", "task_id": task_id},
            ports=[client.V1ServicePort(port=80, target_port=8080)],
            type="ClusterIP",
        ),
    )

    core_v1_api = client.CoreV1Api()
    core_v1_api.create_namespaced_service(namespace=namespace, body=service)
    print(f"Service {service_name} created.")

    # Update the Ingress resource to route traffic to the newly created service
    update_ingress_with_service(task_id, service_name, namespace)

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
                    api_instance.delete_namespaced_job(name=job_name, namespace=namespace, body=client.V1DeleteOptions())
                    break
    except Exception as e:
        print(f"Error while monitoring job status: {e}")
        fl_server_status[task_id]["status"] = "Error"
    finally:
        print("Cleaning up resources")
        w.stop()