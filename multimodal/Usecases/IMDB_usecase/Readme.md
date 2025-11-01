FedOps MMIMDb (Multimodal) Federated Learning Example

This repository contains a multimodal (text + image) federated learning example built on Flower (FLWR) and FedOps, using the MM-IMDb dataset layout.
It provides both client and server entry points, robust data loading (including nested server_data zips), and a simple fusion model (BERT + ResNet18).

Contents

Features

Repository Layout

Prerequisites

Installation

Dataset Layout

Configuration

Run: Server

Run: Client

Model

Troubleshooting

Notes

Citations

License

Features

Text encoder: BERT (HuggingFace Transformers)

Image encoder: ResNet18 (torchvision)

Fusion: concatenation -> FC -> BatchNorm -> ReLU -> classifier

Multilabel loss: BCEWithLogitsLoss

Federated orchestration: Flower + FedOps

Robust labels.json resolution and nested server_data/ handling

Per-client modality masking via modality.json (optional)

Repository Layout

Example key paths (focus on the IMDB use case):

FedOps/
  silo/
    examples/
      torch/
        new_hateful_memes_classification/
          new_hateful_memes/
            FedMAP/
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
                  mmimdb_posters/  (images)
                dataset/
                  client_0/{train.csv,val.csv,test.csv,modality.json?}
                  client_1/{...}
                  ...
                  server/server_test.csv   (optional fast-path for server eval)
                server_data/               (fallback server eval package; can be nested)

Prerequisites

Python 3.9+ (tested)

PyTorch, torchvision

transformers

fastapi, uvicorn

flower

fedops (the FedOps package you are using)

pandas, pillow

Optional: gdown for downloading the tiny server_data.zip

Installation

Create and activate an environment, then install dependencies. Example using conda:

conda create -n fedops_fedmm_env python=3.9 -y
conda activate fedops_fedmm_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA/CPU as needed
pip install transformers==4.* pandas pillow matplotlib fastapi uvicorn
pip install flwr==1.*  # or the version required by FedOps
pip install fedops     # your FedOps package/distribution
pip install gdown      # optional, only if you rely on auto-download of server_data.zip

Dataset Layout

This project expects the MM-IMDb split in CSV form:

all_in_one_dataset/labels.json – canonical list of classes (array of genre strings).

all_in_one_dataset/mmimdb_posters/ – images, file names referenced by CSVs.

dataset/client_{id}/{train.csv,val.csv,test.csv} – per-client splits with columns:

img_name (file name only)

text (plot or description)

labels (pipe-separated e.g. Drama|Romance)

Optional server validation (two ways):

Fast path:

dataset/server/server_test.csv   # uses images from all_in_one_dataset/mmimdb_posters


Fallback path (mounted or auto-downloaded):

server_data/
  server_test.csv
  mmimdb_posters/  or  img/


If your zip unpacks as server_data/server_data/..., the loader automatically normalizes nested folders.

Configuration

Main configuration is in conf/config.yaml. Example:

random_seed: 42
learning_rate: 0.0001
model_type: Pytorch

model:
  _target_: models.MMIMDbFusionModel
  output_size: 23    # must match length of labels.json

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
batch_size: 32
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


Notes:

output_size must equal the number of entries in labels.json.

batch_size: 32 is recommended to avoid BatchNorm errors (see Troubleshooting).

Run: Server

Start your FedOps/Flower server process for this task. Depending on your FedOps deployment the command may differ. Example:

python server_main.py


Server responsibilities here include preparing a small central evaluation loader via:

dataset/server/server_test.csv with project posters (fast path), or

server_data/{server_test.csv, mmimdb_posters|img} (fallback, optionally downloaded via gdown).

Run: Client

In another terminal (per client), run:

python client_main.py


What happens:

Loads client split: dataset/client_{client_id}/

Resolves labels.json robustly

Builds the fusion model

Connects to Flower server and starts federated training

If your server address is discovered via FedOps API, you will see a log like:

FL_server_IP:port - ccl.gachon.ac.kr:40026

Model

Defined in models.py:

Text: BertModel (pooler output, 768d)

Image: ResNet18 (512d) -> Linear to image_output_dim (default 256)

Fusion: concat(text, image) -> Linear(fusion_output_dim) -> BatchNorm -> ReLU -> Classifier(output_size)

Loss: BCEWithLogitsLoss for multilabel

Training helpers (train_torch, test_torch) compute loss and micro-F1 (threshold 0.5).

Troubleshooting
1) labels.json not found

Error:

FileNotFoundError: labels.json not found...


Resolution order in data_preparation.py:

Environment variable LABELS_JSON_PATH (absolute or relative working path)

./all_in_one_dataset/labels.json

<normalized server_data root>/labels.json

First labels.json found recursively under normalized server_data/

Fixes:

Ensure all_in_one_dataset/labels.json exists, or

Place labels.json under server_data/ (or nested), or

Set env: export LABELS_JSON_PATH=/abs/path/to/labels.json

2) Nested server_data after unzip

Symptom: Your zip extracts to server_data/server_data/....
Resolution: The helper _server_root() walks down repeated server_data/ segments automatically. No action needed. Just keep files under that nested folder.

3) BatchNorm error when batch size becomes 1

Error:

ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])


Cause: BatchNorm needs batch size >= 2 in training mode.
Fixes:

Set batch_size: 32 (or at least 2) in conf/config.yaml.

Alternatively, replace BatchNorm1d with LayerNorm or nn.Identity in models.py if you must train with batch size 1.

4) Deprecation warnings from Flower client

Messages about start_numpy_client being deprecated.
Fix: Keep current behavior for now, or migrate to:

flwr.client.start_client(
    server_address="<IP>:<PORT>",
    client=FlowerClient().to_client(),
)


Follow your FedOps version constraints before changing.

5) Vision weights argument change

If you see torchvision warnings, the code already tries both:

models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# and fallback:
models.resnet18(pretrained=True)


Upgrade torchvision if needed, or set pretrained_vision: false.

6) PIL interpolation deprecations

Warnings about BILINEAR/NEAREST/BICUBIC in PIL. They are warnings only.
To silence, you can switch to the new Resampling enum in your transforms.

Notes

GPU strongly recommended for BERT + ResNet18.

If your images are missing or some paths are broken, data_preparation falls back to zero image tensors to keep batches flowing, but training quality will drop.

Modality masking: put a modality.json in dataset/client_{id}/ such as:

{ "use_text": 1, "use_image": 0 }


to zero-out image features for that client.

Citations

Flower: https://flower.dev/

HuggingFace Transformers: https://github.com/huggingface/transformers

torchvision: https://pytorch.org/vision/stable/index.html

MM-IMDb dataset: https://github.com/ashudeep/Multimodal-Movie-Genre-Classification
 (original reference)

License

This example is provided under your project’s license.
Check the licenses of Flower, FedOps, Transformers, and torchvision before redistribution
