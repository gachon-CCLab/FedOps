FedMAP: Multimodal Federated Learning on MM-IMDb (BERT + ResNet)

Federated multimodal learning built on FedOps + Flower, using BERT (text) and ResNet-18 (image) with a fusion head for multilabel genre prediction on MM-IMDb.

Highlights

Model: BERT (text) + ResNet-18 (image) → Fusion MLP → multilabel classifier (BCEWithLogitsLoss)

FL Stack: Flower orchestration with FedOps client/server, Modality-Aware Aggregation (FedMAP)

Robust I/O: Safe labels.json resolution; handles nested server_data/ after unzip

Hydra Configs: Reproducible runs with simple overrides

Pitfall Fixes: BatchNorm w/ small batches, Pillow/TV image warnings

Repository Layout
<repo-root>/
├─ client_main.py                # FL client entry (FastAPI + Flower client)
├─ server_main.py                # FL server entry (FedMAP strategy)
├─ data_preparation.py           # Datasets, loaders, path resolvers (client+server)
├─ models.py                     # MMIMDbFusionModel + train/test loops
├─ flclient_patch.py             # Client subclass hook (metrics/patches)
├─ conf/
│  └─ config.yaml                # Hydra config (batch_size, rounds, etc.)
├─ all_in_one_dataset/
│  ├─ mmimdb_posters/            # Poster images
│  └─ labels.json                # Canonical labels (C=23)
├─ dataset/
│  ├─ client_0/{train,val,test}.csv
│  ├─ client_1/{...}.csv
│  └─ server/server_test.csv     # (optional) fast-path server eval CSV
└─ server_data/                  # (optional) server eval (may unzip nested)
   ├─ server_data.zip
   └─ ... (possibly server_data/server_data/...)

Installation
conda create -n fedops_fedmm_env python=3.9 -y
conda activate fedops_fedmm_env
pip install -r requirements.txt


Requirements include: PyTorch/TorchVision, Transformers, Flower, FedOps, Hydra, FastAPI, Uvicorn, pandas, Pillow, (optional) gdown.

Data Preparation
1) Client CSVs

Each CSV row must have:

img_name – image filename present in posters folder

text – movie plot/summary

labels – pipe-separated labels, e.g. Drama|Romance

Place under:

./dataset/client_0/{train,val,test}.csv
./dataset/client_1/{train,val,test}.csv
...

2) Posters & Labels
./all_in_one_dataset/mmimdb_posters/   # images
./all_in_one_dataset/labels.json       # ["Action", "Comedy", ...] (C=23)


Alternative: set LABELS_JSON_PATH=/abs/path/to/labels.json.

3) Server Validation (optional)

Fast path: ./dataset/server/server_test.csv

Fallback: place server_test.csv and mmimdb_posters/ or img/ under ./server_data/
(Auto-download from Drive if gdown available and files missing.)

If unzip creates server_data/server_data/..., this project auto-descends to the true root.

Configuration (Hydra)

conf/config.yaml (example):

random_seed: 42
learning_rate: 0.0001

model_type: Pytorch
model:
  _target_: models.MMIMDbFusionModel
  output_size: 23            # must match len(labels.json)

dataset:
  name: MM_IMDB
  validation_split: 0.0

client_id: 0
task_id: akeelcustomusecase

num_epochs: 1
batch_size: 32               # keep ≥2 to avoid BN issues
num_rounds: 2
clients_per_round: 1

server:
  strategy:
    _target_: fedops.server.fedmap.strategy.ModalityAwareAggregation
    aggregator_path: "aggregator_mlp.pth"
    input_dim: 10
    hidden_dim: 16
    aggregator_lr: 0.001
    entropy_coeff: 0.01
    n_trials_per_round: 4
    perf_mix_lambda: 0.7
    z_clip: 3.0


Override at runtime, e.g.:

python client_main.py batch_size=64 client_id=1 num_epochs=2

Running
Start Server
python server_main.py

Start Client(s)
python client_main.py


Client discovers server IP:port via FedOps and connects to Flower.

Expect deprecation warnings from Flower if using legacy start APIs (see below).

Model Details

MMIMDbFusionModel

Text: BertModel.from_pretrained("bert-base-uncased") → pooler_output (768-d)

Image: ResNet-18 (ImageNet weights) → 512-d → Linear(512 → image_output_dim)

Fusion: cat([text,image]) → Linear → BatchNorm1d → ReLU → Linear(C)

Loss: BCEWithLogitsLoss (multilabel)

Metric: micro-F1 @ 0.5 (reported in train/test)

Troubleshooting
BatchNorm error on small batches
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])


Fix options:

Use batch_size: 32 (what worked for you), or ensure every training batch has ≥2 samples.

Set drop_last=True for the train DataLoader.

Replace fusion BatchNorm1d with LayerNorm/GroupNorm:

# self.fusion_bn = nn.BatchNorm1d(fusion_output_dim)
self.fusion_bn = nn.LayerNorm(fusion_output_dim)

labels.json not found

Resolved in this order:

LABELS_JSON_PATH (env; absolute or relative)

./all_in_one_dataset/labels.json

<normalized server_data root>/labels.json

First labels.json found under server_data/

Ensure it exists in one of the above.

Nested server_data after unzip

No action needed; the code auto-walks down nested server_data/server_data/....

Notes on Flower Deprecations

You may see warnings about start_numpy_client() / start_client() being deprecated.
The example remains functional; to modernize, switch to flower-supernode when convenient:

flower-supernode --insecure --superlink='<IP>:<PORT>'

Requirements

Python 3.9/3.10

PyTorch, TorchVision

Transformers (HuggingFace)

Flower, FedOps

Hydra, FastAPI, Uvicorn

pandas, Pillow

(Optional) gdown (for auto-download of server data)

Install:

pip install -r requirements.txt
