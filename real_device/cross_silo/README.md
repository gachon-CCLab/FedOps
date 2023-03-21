# Client(PC) in Federated Learning Enviroment

## Run client (Docker ver.)
- **Make sure you have installed docker compose environment**  
You can run with **docker desktop** or **docker engine + docker compose**.  
Install: (https://www.docker.com/products/docker-desktop/)  
Install docs: (https://docs.docker.com/desktop/)  
Reboot after initial installation of docker.

- **clone this git repo**
```bash
# clone git repository
$ git clone https://github.com/gachon-CCLab/fl-client.git
$ cd fl-client
```

- **run docker compose**
```bash
# run docker compose
$ docker compose up

# or if you want to monitor client, run below
$ docker compose -f docker-compose-monitoring.yml up
```
When you want to apply code changes, do `docker compose up --build`.  


- **terminate client**
```bash
# press Ctrl + C to escape
$ docker compose down
```   

*(Tested on Docker Desktop 4.15.0; Docker Compose version 2.13.0; arm64 processor(macOS))*   
*(Tested on x86-64 processor)*


## Run client (shell ver.)
### **Directory configuration**

- Client Dir
    - client.py: FL Client Code (Data & Model Setting)
    - client_utils.py: client management code

- Client Manager Dir
    - client_manager.py: client status & server status management(via API Communication)

- client.sh: client.py execution shell
- client_manager.sh: client_manager.py execution shell
- requirments.txt: Libraries that require installation

### **Pre-preparation before running**

- **Clone this Git Repo**
- **Create a new conda environment or use an existing one**

```bash
# create conda enviroment
conda create -n fl

# activate conda enviroment 
conda activate fl

# or use an existing one
conda activate existing your conda enviroment
```

- **Install the requirements in your pc(conda environment)**

```bash
# Go to clone repo
pip install -r requirements.txt
```

### Start FL Client

- **Create two CLI and run shell file**

```bash
# One CLI => client
sh client.sh

# The other CLI => client_manager
sh client_manager.sh

```

## Next Work

- FL Server의 Config 값을 Server-Status에 전달할 수 있도록 구성

```python
num_rounds = 10
local_epochs = 5
batch_size = 128
val_steps = 10
```

```python
fraction_fit=1.0,  # 클라이언트 학습 참여 비율
fraction_evaluate=1.0,  # 클라이언트 평가 참여 비율
min_fit_clients=5,  # 최소 학습 참여 수
min_evaluate_clients=5,  # 최소 평가 참여 수
min_available_clients=5,  # 최소 클라이언트 연결 필요 수
```

- 현재는 cifar 10 data와 model을 수행하였지만, 다양한 data/model 구성이 가능하도록 구성
    - FL Server도 초기에 global model을 생성하기 때문에 client에서 초기 local_model을 전달하는 방법을 고안
    - FL Server global model 평가를 위한 dataset은 어떻게 할지 구성할지 고안
