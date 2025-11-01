FedMAP – Multimodal Federated Learning on MM-IMDb (BERT + ResNet)

Federated multimodal learning example built on FedOps + Flower, using BERT for text and ResNet-18 for images, fused with a small MLP head for multilabel movie-genre prediction on MM-IMDb.

This repo includes:

Robust data preparation that works for both server and clients

A modality-aware aggregation strategy (FedMAP) on the server

A PyTorch fusion model (BERT + ResNet-18 → FC → Norm → Classifier)

Hydra configs for clean experiments

Fixes for common pitfalls (e.g., labels.json resolution, nested server_data unzips, batch-norm with tiny batches)

TL;DR (Quickstart)
# 1) Create env
conda create -n fedops_fedmm_env python=3.9 -y
conda activate fedops_fedmm_env

# 2) Install deps
pip install -r requirements.txt
# (FedOps, Flower, Transformers, TorchVision, FastAPI, Uvicorn, Hydra, pandas, Pillow, etc.)

# 3) Prepare data (see “Data Layout” below)
# - ./all_in_one_dataset/mmimdb_posters
# - ./all_in_one_dataset/labels.json
# - ./dataset/client_0/{train,val,test}.csv (and other clients)
# - Optional server eval set under ./server_data/...

# 4) Start FL server (FedOps/Flower)
python server_main.py  # runs strategy = ModalityAwareAggregation

# 5) Start one or more clients
python client_main.py  # uses Hydra ./conf/config.yaml (e.g., batch_size=32)

Features

Multimodal architecture: Text via BertModel (pooler output), image via ResNet-18 → fused embedding → classifier (multilabel with BCEWithLogitsLoss).

Federated training: Flower-based orchestration with FedOps client wrapper and ModalityAwareAggregation server strategy.

Robust I/O:

labels.json is auto-resolved from multiple locations or env var.

Handles nested server_data/server_data/... after unzip.

Optional per-client modality masking (modality.json).

Hydra configs for reproducible experiments.

Repo Layout (key pieces)
<repo-root>/
├─ server_main.py                   # FL server entry
├─ client_main.py                   # FL client entry (FastAPI + Flower client)
├─ data_preparation.py              # Datasets, loaders, robust path resolvers
├─ models.py                        # MMIMDbFusionModel + train/test loops
├─ flclient_patch.py                # Client subclass hook for custom logic
├─ conf/
│  └─ config.yaml                   # Hydra config (batch size, rounds, etc.)
├─ all_in_one_dataset/
│  ├─ mmimdb_posters/               # Poster images (project-wide)
│  └─ labels.json                   # Canonical genre label list (C=23)
├─ dataset/
│  ├─ client_0/{train,val,test}.csv # Client partitions
│  ├─ client_1/{...}.csv
│  └─ server/server_test.csv        # (optional) fast-path server eval CSV
└─ server_data/                      # (optional) server eval zip/unzipped data
   ├─ server_data.zip               # auto-downloaded if missing (requires gdown)
   └─ ... (may nest as server_data/server_data/...)

Data Layout
1) Client splits

CSV columns expected by MMIMDbDataset:

img_name – filename in the posters folder

text – the movie plot/summary

labels – pipe-separated labels, e.g. Drama|Romance

Place them under:

./dataset/client_0/{train,val,test}.csv
./dataset/client_1/{train,val,test}.csv
...

2) Posters & labels
./all_in_one_dataset/mmimdb_posters/   # images
./all_in_one_dataset/labels.json       # list of class names, e.g. ["Action","Comedy",...]

3) Server validation (optional)

Fast path (if you have it):

./dataset/server/server_test.csv


Otherwise, mount/download into:

./server_data/
  server_data.zip
  server_test.csv
  mmimdb_posters/ or img/


We added a nested server_data fixer: if your zip expands to server_data/server_data/..., the code auto-walks down to the real root.

Configuration (Hydra)

conf/config.yaml (typical fields):

random_seed: 42
learning_rate: 0.0001

model_type: Pytorch
model:
  _target_: models.MMIMDbFusionModel
  output_size: 23            # IMPORTANT: set to len(labels.json)

dataset:
  name: MM_IMDB
  validation_split: 0.0

client_id: 0
task_id: akeelcustomusecase

num_epochs: 1
batch_size: 32               # Use 32+ to avoid BN-on-single-sample issues
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


You can also set LABELS_JSON_PATH=/abs/path/to/labels.json if you keep it elsewhere.

Running
Server
python server_main.py


Uses ModalityAwareAggregation from FedOps.

Exposes a port that clients fetch dynamically.

Client(s)
python client_main.py


Spins up a small FastAPI to coordinate with FedOps/Flower.

Prints the discovered ccl.gachon.ac.kr:<port> endpoint when ready.

Starts Flower NumPyClient (deprecation warnings are expected if you’re on recent Flower; see “Notes & Deprecations”).

Model

MMIMDbFusionModel

Text: BertModel.from_pretrained("bert-base-uncased") → pooler_output (768-d)

Image: torchvision.models.resnet18(weights=IMAGENET1K_V1) → 512-d → Linear(512→image_output_dim)

Fusion: concat([text, image]) → Linear(fused→fusion_output_dim) → Norm → ReLU → Linear(fusion_output_dim→C)

Loss: BCEWithLogitsLoss (multilabel)

Metrics (train/test): micro-F1 (threshold 0.5)

We default to BatchNorm1d in fusion. If you frequently hit micro-batches (B=1), see “Troubleshooting”.

Troubleshooting
1) labels.json not found

We resolve labels.json from:

LABELS_JSON_PATH (env var)

./all_in_one_dataset/labels.json

<normalized server_data root>/labels.json

First labels.json found recursively under <server_data>

Fix: ensure it exists in one of the above, or set the env var:

export LABELS_JSON_PATH=/abs/path/to/labels.json

2) Nested server_data after unzip

Some zips create server_data/server_data/.... We detect and descend automatically. Just keep everything under ./server_data/, and you’re good.

3) ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])

BatchNorm needs at least 2 samples per batch in training mode. You saw this when your last batch size was 1.

Options:

Recommended (what you did): set batch_size: 32 (or any value that keeps batches ≥2).

Or drop the last incomplete batch (only train loader):

DataLoader(..., drop_last=True)


Or switch to LayerNorm/GroupNorm in models.py:

# self.fusion_bn = nn.BatchNorm1d(fusion_output_dim)
self.fusion_bn = nn.LayerNorm(fusion_output_dim)  # robust to B=1

4) Flower deprecation warnings

Recent Flower versions deprecate start_numpy_client() and start_client() in favor of supernode. This example still works but prints warnings. To modernize, migrate to flower-supernode CLI when ready.

Repro Tips

Hydra makes it easy to override parameters:

python client_main.py batch_size=64 num_epochs=2 client_id=1


Fix seeds with random_seed (we already set NumPy/PyTorch seeds).

Log your exact commit + conf/config.yaml with each run.

Requirements

Python 3.9/3.10

PyTorch, TorchVision, Transformers

Flower, FedOps

Hydra, FastAPI, Uvicorn

pandas, Pillow

(Optional) gdown if you want auto-download for server_data.zip

Install with:

pip install -r requirements.txt

Acknowledgements

MM-IMDb dataset by Arevalo et al.

Flower team for the FL framework.

HuggingFace Transformers and TorchVision for models.

FedOps for orchestration utilities and strategy hooks.
