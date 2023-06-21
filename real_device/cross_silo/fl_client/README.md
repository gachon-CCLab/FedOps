# Real Device in Federated Learning Environment

## FL Client Code Structure
### client Dir
**config.yaml**:
- Task ID: Set the task_id to the desired task ID.
- Data: Specify the data settings:
  - name: Set the data name for using to FL task
  - label_count: Set the data label number
  - validation_split
- WandB: Provide the WandB information:
  - api_key: Set the api_key to the desired your WandB.
  - account: Set the WandB's account.

**client_data.py**:
- Set the dataset to be used by the client

**client.model.py**:
- Define the structure of the client's local model

**client_task.py**:
- Register own data & model and run the client in ***client_task.py***

### Client Manager Dir
**client_manager.py**: client status & server status management(via API Communication)



## Run client (Docker ver.)
- **Make sure you have installed docker compose environment**  
You can run with **docker desktop** or **docker engine + docker compose**.  
Install: (https://www.docker.com/products/docker-desktop/)  
Install docs: (https://docs.docker.com/desktop/)  
Reboot after initial installation of docker.

- **run docker compose**
```bash
# run docker-compose.yaml file
$ docker-compose -f docker-compose.yml up -d --build
```


- **terminate client**
```bash
# press Ctrl + C to escape
$ docker compose down
```   

*(Tested on Docker Desktop 4.15.0; Docker Compose version 2.13.0; arm64 processor(macOS))*   
*(Tested on x86-64 processor)*

## Run client (Localhost ver.)
```bash
# run client and client manager to background env
$ sh run_shell_client.sh
```
or
```bash
# run client 
$ sh python client/client_task.py

# run client manager
$ sh python client_manager/client_manager.py
```  
