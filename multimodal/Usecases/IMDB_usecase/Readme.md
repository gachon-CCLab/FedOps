

````markdown
# FedOps MMIMDb (Multimodal) Federated Learning

Multimodal (text + image) FL example built on **Flower (FLWR)** and **FedOps** using an MM-IMDb–style dataset.  
This utilizes fedmap strategy.And includes robust `server_data/` handling (even when nested) and a simple BERT + ResNet18 fusion model.

---

## Table of Contents

- [Features](#features)
- [Repo Layout](#repo-layout)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Dataset Layout](#dataset-layout)
- [Configuration](#configuration)
- [Run: Server](#run-server)
- [Run: Client](#run-client)
- [Model](#model)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

---

## Features

- Text: BERT (Transformers)
- Image: ResNet18 (torchvision)
- Fusion: concat → FC → BatchNorm1d → ReLU → classifier
- Loss: `BCEWithLogitsLoss` (multilabel)
- Orchestration: Flower + FedOps
- Robust `labels.json` discovery
- Optional per-client modality masking (`modality.json`)

---

## Repo Layout


IMDB_usecase/
                client_main.py
                server_main.py
                data_preparation.py
                models.py
                flclient_patch.py
                conf/
                  config.yaml
                all_in_one_dataset/
                  labels.json
                  mmimdb_posters/
                dataset/
                  client_0/{train.csv,val.csv,test.csv,modality.json?}
                  client_1/{...}
                  
                
````

---

## Prerequisites

* Python 3.9+
* PyTorch, torchvision
* transformers
* fastapi, uvicorn
* flwr
* fedops
* pandas, pillow
* Optional: `gdown` (for tiny server_data zip)

---

## Install

```bash
conda create -n fedops_fedmm_env python=3.9 -y
conda activate fedops_fedmm_env

# Choose the correct CUDA/CPU wheel for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install "transformers==4.*" pandas pillow matplotlib fastapi uvicorn
pip install "flwr==1.*"
pip install fedops
pip install gdown   # optional, only if you rely on auto-download
```

---

## Dataset Layout

```
all_in_one_dataset/
  labels.json            # array of class names
  mmimdb_posters/        # images

dataset/
  client_0/{train.csv,val.csv,test.csv}
  client_1/{...}
  server/server_test.csv # (fast path for server eval)

```

CSV columns expected: `img_name`, `text`, `labels` (pipe-separated, e.g., `Drama|Romance`).

> If your zip creates `server_data/server_data/...`, the loader auto-normalizes it.

---

## Configuration

`conf/config.yaml` (example):

```yaml
random_seed: 42
learning_rate: 0.0001
model_type: Pytorch

model:
  _target_: models.MMIMDbFusionModel
  output_size: 23           # must equal len(all_in_one_dataset/labels.json)

dataset:
  name: MM_IMDB
  validation_split: 0.0

client_id: 0
task_id: akeelcustomusecase

wandb:
  use: false
  key: your wandb api key
  account: your wandb account
  project: ${dataset.name}_${task_id}

num_epochs: 1
batch_size: 32             # >=2 to avoid BatchNorm errors
num_rounds: 2
clients_per_round: 1

server:
  strategy:
    _target_: fedops.server.fedmap.strategy.ModalityAwareAggregation
    aggregator_path: aggregator_mlp.pth
    input_dim: 10
    hidden_dim: 16
    aggregator_lr: 0.001
    entropy_coeff: 0.01
    n_trials_per_round: 4
    perf_mix_lambda: 0.7
    z_clip: 3.0
```

---

## Run: Server


```
To run server go to Fedops task UI and follow the intructions given in thsi website:http://210.102.181.208:40007/document/690427da788e28e19c8b2b9b
The server prepares central eval from either:

* `dataset/server/server_test.csv` + project posters (fast path), or
* `server_data/{server_test.csv, mmimdb_posters|img}` (fallback; can auto-download with `gdown`).

---

## Run: Client

```bash
python client_main.py
```

You should see logs like:

```
FL_Task - {"dataset": "MM_IMDB", "client_id": 0}
FL_server_IP:port - ccl.gachon.ac.kr:40026
```

---

## Model

* **Text**: `BertModel` → pooler (768d)
* **Image**: `ResNet18` (512d) → `Linear(256)`
* **Fusion**: `[text, image]` concat → `Linear(256)` → `BatchNorm1d` → `ReLU` → `Linear(output_size)`
* **Loss/Metric**: `BCEWithLogitsLoss`; micro-F1 (threshold 0.5)

---

## Troubleshooting

### `labels.json not found`

Resolution order:

1. `LABELS_JSON_PATH`
2. `./all_in_one_dataset/labels.json`
3. normalized `server_data/labels.json`
4. first recursive match under normalized `server_data/`.

Fix by ensuring one of the above exists or exporting:

```bash
export LABELS_JSON_PATH=/abs/path/to/labels.json
```

### BatchNorm error with batch size 1

Error:

```
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])
```

Fix: set `batch_size >= 2` (e.g., 32), or replace `BatchNorm1d` with `LayerNorm`/`Identity`.



### Flower deprecations

You may see warnings about `start_numpy_client`. Keep current FedOps integration or migrate to:

```python
flwr.client.start_client(
  server_address="<IP>:<PORT>",
  client=FlowerClient().to_client(),
)
```

---

## Notes

* GPU recommended (BERT + ResNet18).
* Missing images are auto-zeroed to keep batches flowing (training quality will drop).
* Optional per-client modality mask in `dataset/client_{id}/modality.json`:

  ```json
  { "use_text": 1, "use_image": 0 }
  ```

---


