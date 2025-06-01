````markdown
# Multimodal Federated Hateful Memes Classification (FedOps Silo)

This repository demonstrates a full end-to-end **multimodal** (text + image)
federated learning (FL) setup using FedOps Silo and Flower, applied to the
[Hateful Memes](https://huggingface.co/neuralcatcher/hateful_memes) dataset.
Each client trains a fusion model on its local (non-IID) split of the memes,
and a centralized FL server aggregates updates across all clients.

---

## 🚀 Features

- **Multimodal Fusion (Early, Intermediate, Late)**  
  1. **Early Fusion**: Concatenate text embeddings and image embeddings
     before deeper processing.  
  2. **Intermediate Fusion**: Use cross-attention to let one modality guide
     the other (e.g. image attends over text features).  
  3. **Late Fusion**: Train separate unimodal classifiers for text and image,
     then combine their output logits or probabilities in a final decision
     layer.

  The default `HatefulMemesFusionModel` uses simple concatenation of BERT
  pooled output and CNN image features. Custom fusion variants live under
  `multimodal/FusionModel/`.

- **Federated Learning Stack**  
  Uses [FedOps Silo](https://github.com/fedops-fedmm) and [Flower](https://flower.dev):
  1. **Clients** run local PyTorch training loops (text + image).  
  2. **Server** uses a Flower Strategy (e.g. FedAvg or a custom aggregator)
     to merge client weights each round.

- **Non-IID Data Splits**  
  Each client receives a stratified, non-IID subset of the Hateful Memes
  dataset. A hold-out validation set remains with the server for global
  evaluation.

- **GPU Acceleration**  
  Clients detect and utilize any available GPU for faster model training
  (CUDA + cuDNN).

- **Custom Aggregation Strategies**  
  Easily add your own FL Strategy (e.g. modality-aware weight fusion) under
  `multimodal/Aggregation/` and point to it in `conf/config.yaml`.

- **Hydra Configuration**  
  All hyperparameters, model settings, and FL server/strategy parameters live
  in a single `conf/config.yaml` file that works for both server & client.

---

## 📂 Folder Structure

```plaintext
Hateful_memes_classification/
├── clienta_main.py            # “Client A” entry point (GPU setup + FL client)
├── clientb_main.py            # “Client B” entry point (similar to clienta_main.py)
├── client_manager_main.py     # Flower client manager (registers client
│                              # with central coordinator)
├── server_main.py             # FL server entry point (starts Flower server)
├── models.py                  # Defines `HatefulMemesFusionModel`
│                              # (text+image fusion)
├── data_preparation.py        # Loads Hateful Memes dataset,
│                              # returns PyTorch DataLoaders
├── multimodal/
│   ├── FusionModel/
│   │   └── custom_fusion.py   # (optional) alternative fusion architectures
│   └── Aggregation/
│       └── my_custom_strategy.py  # Example custom Flower Strategy subclass
├── conf/
│   └── config.yaml             # Hydra config (model, FL server, strategy,
│                              # hyperparams)
├── requirements.txt            # Python dependencies
└── README.md                   # ← You are reading this file
````

---

## 🔧 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Hateful_memes_classification.git
cd Hateful_memes_classification
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure your CUDA drivers and PyTorch CUDA build are correctly installed if
you intend to use a GPU.

---

### 2. Configuration (`conf/config.yaml`)

All hyperparameters and file paths live in one place. Example:

```yaml
# conf/config.yaml

random_seed: 42
lr: 0.0001

model_type: 'Pytorch'
model:
  _target_: models.HatefulMemesFusionModel
  text_hidden_dim: 768
  image_output_dim: 128
  fusion_output_dim: 256
  output_size: 2

dataset:
  name: 'hateful_memes'
  validation_split: 0.2  # 20% held out at server

task_id: 'hatetaskthree'

wandb:
  use: false

# FL setup
num_epochs: 1
batch_size: 32
num_rounds: 2
clients_per_round: 2

server:
  strategy:
    _target_: multimodal.Aggregation.my_custom_strategy.MyCustomStrategy
    alpha: 0.7
    fraction_fit: 0.00001
    fraction_evaluate: 0.000001
    min_fit_clients: ${clients_per_round}
    min_available_clients: ${clients_per_round}
    min_evaluate_clients: ${clients_per_round}
```

* `_target_` under `model:` points to the PyTorch fusion model class in
  `models.py` (or a custom fusion file under `multimodal/FusionModel/`).
* `_target_` under `server.strategy:` can be Flower’s built-in `FedAvg` or
  your own class (e.g.
  `multimodal.Aggregation.my_custom_strategy.MyCustomStrategy`).

---

### 3. Non-IID Data Splitting

Each client loads its own **local** (non-IID) partition of the Hateful Memes
dataset. By default, `data_preparation.py` uses the **entire** dataset in each
client, but you can customize:

```python
# data_preparation.py

def load_partition(batch_size=32, split="train", client_id=None):
    full_dataset = load_dataset(
        "neuralcatcher/hateful_memes", split=split
    )
    if client_id is not None:
        # Example: stratified 50/50 split by “label”
        indices = stratified_indices_for_client(
            full_dataset["label"], client_id, num_clients=2
        )
        local_subset = full_dataset.select(indices)
    else:
        local_subset = full_dataset

    # … tokenize, transform images, return DataLoader(local_subset) …
```

Implement `stratified_indices_for_client(...)` under `data_preparation.py` and
pass `client_id=0` or `1` when each client calls `load_partition(...)`.

---

### 4. Running Clients & Server

#### 4.1 Start the Server

```bash
python server_main.py --config-name=config
```

This spins up the Flower server on `0.0.0.0:8787` by default and waits for
clients.

#### 4.2 Start Client Manager (per client)

On **Client A** machine:

```bash
python client_manager_main.py
```

This FastAPI manager polls the server for “start training” signals and
forwards them to the local FL client.

#### 4.3 Start FL Client (per client)

On **Client A**:

```bash
python clienta_main.py --config-name=config
```

* Client prints: `🔌 Using device: cuda` if CUDA is available, else `cpu`.
* Client loads its local (non-IID) split via
  `data_preparation.load_partition(...)`.
* Client manager signals “start training” once both clients register online.
* Flower’s `start_numpy_client` starts under the hood and begins local
  training.

On **Client B** (another machine or terminal):

```bash
python clientb_main.py --config-name=config
```

---

## 🖥️ GPU Acceleration (Client Side)

Each client’s main file includes:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
```

The fusion model is moved onto that device:

```python
model = instantiate(cfg.model)
model = model.to(device)
```

In `models.py`, train and test loops send tensors to `device` with
`batch["…"].to(device, non_blocking=True)`. For multi-GPU, set
`CUDA_VISIBLE_DEVICES=0,1` when launching the client.

---

## ⚙️ Custom Aggregation Strategy

1. **Create a file** under `multimodal/Aggregation/my_custom_strategy.py`:

   ```python
   # multimodal/Aggregation/my_custom_strategy.py

   from typing import List, Tuple
   from flwr.server.strategy import FedAvg
   from flwr.server.client_proxy import ClientProxy
   from flwr.common import FitRes

   class MyCustomStrategy(FedAvg):
       def __init__(self, alpha: float = 0.5, **kwargs):
           super().__init__(**kwargs)
           self.alpha = alpha

       def aggregate_fit(
           self,
           rnd: int,
           results: List[Tuple[ClientProxy, FitRes]],
           failures: List[BaseException],
       ):
           """
           Example: Weight client updates by a combination of validation accuracy
           and local sample size.
           """
           weighted_results = []
           total_samples = sum(
               res.metrics["num_examples"] for _, res in results
           )

           for client, fit_res in results:
               local_samples = fit_res.metrics["num_examples"]
               local_acc = fit_res.metrics.get("accuracy", 0.0)
               w_size = local_samples / total_samples
               weight_factor = (
                   self.alpha * local_acc + (1.0 - self.alpha) * w_size
               )
               weighted_results.append((client, fit_res, weight_factor))

           return super().aggregate_fit(
               rnd,
               [
                   (client, fit_res, weight_factor)
                   for client, fit_res, weight_factor in weighted_results
               ],
               failures,
           )
   ```

2. **Update `conf/config.yaml`** to point to this class:

   ```yaml
   server:
     strategy:
       _target_:
         multimodal.Aggregation.my_custom_strategy.MyCustomStrategy
       alpha: 0.7
       fraction_fit: 0.00001
       fraction_evaluate: 0.000001
       min_fit_clients: ${clients_per_round}
       min_available_clients: ${clients_per_round}
       min_evaluate_clients: ${clients_per_round}
   ```

Hydra will instantiate your custom strategy at server startup.

---

## 🔄 Custom Model Fusion

By default, `models.py` implements simple concatenation (Early Fusion). To
experiment with other fusion types:

1. **Early Fusion** (default): Concatenate BERT pooled output and CNN image
   features.

2. **Intermediate Fusion** (Cross-Attention): Place a new file under
   `multimodal/FusionModel/custom_fusion.py`, e.g.:

   ```python
   # multimodal/FusionModel/custom_fusion.py

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from transformers import BertModel

   class CrossAttentionFusionModel(nn.Module):
       def __init__(
           self,
           text_hidden_dim=768,
           image_output_dim=128,
           fusion_output_dim=256,
           output_size=2,
           num_heads=8,
       ):
           super().__init__()
           self.bert = BertModel.from_pretrained("bert-base-uncased")
           self.image_encoder = nn.Sequential(
               nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2),
               nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d((4, 4)),
           )
           self.flatten = nn.Flatten()
           self.image_fc = nn.Linear(64 * 4 * 4, image_output_dim)

           # Cross-Attention block
           self.cross_attention = nn.MultiheadAttention(
               embed_dim=text_hidden_dim, num_heads=num_heads
           )
           self.fusion_fc = nn.Linear(text_hidden_dim + image_output_dim,
                                      fusion_output_dim)
           self.classifier = nn.Linear(fusion_output_dim, output_size)

       def forward(self, input_ids, attention_mask, image):
           bert_out = self.bert(
               input_ids=input_ids, attention_mask=attention_mask
           )
           text_feat = bert_out.pooler_output  # (batch, 768)
           text_feat = text_feat.unsqueeze(1)  # (batch, 1, 768)

           img_feat = self.image_encoder(image)  # (batch, 64, 4, 4)
           img_feat = self.flatten(img_feat)      # (batch, 64*4*4)
           img_feat = self.image_fc(img_feat)     # (batch, 128)
           img_feat = img_feat.unsqueeze(0)       # (1, batch, 128)

           # Cross-attend image queries over text keys/values
           text_kv = text_feat.transpose(0, 1)  # (1, batch, 768)
           attn_out, _ = self.cross_attention(
               query=img_feat,
               key=text_kv,
               value=text_kv,
           )
           attn_out = attn_out.squeeze(0)  # (batch, 768)

           fused = torch.cat((attn_out, img_feat.squeeze(0)), dim=1)
           fusion_out = F.relu(self.fusion_fc(fused))
           logits = self.classifier(fusion_out)
           return logits
   ```

3. **Late Fusion**: In `models.py`, implement separate text-only and image-only
   heads, then combine:

   ```python
   class LateFusionModel(nn.Module):
       def __init__(self, ...):
           super().__init__()
           self.bert = BertModel.from_pretrained("bert-base-uncased")
           self.image_encoder = ...
           self.text_head = nn.Linear(text_hidden_dim, output_size)
           self.image_head = nn.Linear(image_output_dim, output_size)
           self.late_fusion_layer = nn.Linear(output_size * 2, output_size)

       def forward(self, input_ids, attention_mask, image):
           text_feat = self.bert(...).pooler_output  # (batch, 768)
           text_logits = self.text_head(text_feat)   # (batch, 2)

           img_feat = self.image_encoder(image)      # ...
           img_feat = self.image_fc(img_feat)        # (batch, 128)
           img_logits = self.image_head(img_feat)    # (batch, 2)

           fused_logits = torch.cat([text_logits, img_logits], dim=1)
           final_logits = self.late_fusion_layer(fused_logits)
           return final_logits
   ```

To switch, update `_target_` under `model:` in `conf/config.yaml` to the new
class path.

---

## 📖 Usage Workflow

1. **(Optional) Kubernetes**
   Build Docker images for:

   * FL server (includes `server_main.py` + dependencies)
   * Each client (includes `client_manager_main.py` + `clienta_main.py` or
     `clientb_main.py`)

   Ensure pods have GPU resources (`resources.requests.limits`). If pods get
   evicted due to `out-of-memory` or `out-of-shared-memory`, increase pod
   memory limits or use a larger node pool.

2. **Launch FL Server**

   ```bash
   python server_main.py --config-name=config
   ```

   * Server loads a **hold-out validation set** (20% of full data) via
     `data_preparation.gl_model_torch_validation`.
   * Starts Flower on `0.0.0.0:8787` (default).

3. **Launch Client Manager**

   ```bash
   python client_manager_main.py
   ```

   * FastAPI manager polls the server at `/FLSe/info` every few seconds.
   * When the server signals “start training,” the manager forwards a `/start`
     request to the FL client.

4. **Launch FL Client**

   ```bash
   python clienta_main.py --config-name=config
   ```

   * Prints `🔌 Using device: cuda` if CUDA is available.
   * Loads local (non-IID) subset via `data_preparation.load_partition(...)`.
   * Waits for manager’s “start training” signal, then spins up Flower’s client.
   * Logs local train/validation metrics each round.

   Repeat on **Client B**:

   ```bash
   python clientb_main.py --config-name=config
   ```

5. **Monitoring & Logs**

   * Server logs round-by-round metrics: initial validation loss/accuracy, then
     after each aggregation.
   * Clients log local train/validation metrics.
   * If a client times out or loses connection, you’ll see a “ping timeout” or
     `StatusCode.UNAVAILABLE`; the manager marks it offline and updates the
     server.

6. **Hold-Out Validation on Server**

   * The server’s `FLServer` uses `gl_val_loader` (20% hold-out) to compute
     global model performance each round, independent of client validation.

---

## 📚 Extending & Publishing

1. **Add New Aggregation Strategies**

   * Create new strategy classes under `multimodal/Aggregation/`.
   * Update `_target_` in `conf/config.yaml`.
   * Run `python server_main.py` to test locally.

2. **Publish to PyPI**

   * Add `setup.py` or `pyproject.toml` at project root.

   * Under `packages=[ "multimodal.Aggregation", … ]` list your modules.

   * In `install_requires=[ … ]`, list needed dependencies.

   * Then:

     ```bash
     python3 -m build
     twine upload dist/*
     ```

   * Others can `pip install your-package` and point `_target_` to
     `your_package.my_custom_strategy.MyCustomStrategy`.

3. **Contribute to FedOps / Flower**
   If your aggregator outperforms existing multimodal strategies, consider
   opening a PR or writing a tutorial on Flower’s site.

---

## ✨ Acknowledgments

* **FedOps Silo** – Federated learning boilerplate for PyTorch & Flower.
* **Flower** – A lightweight federated learning framework.
* **Hateful Memes** – A multimodal dataset (text + images) for offensive content
  classification.
* **HuggingFace transformers** – BERT model for text encoding.
* **PyTorch** – Deep learning library powering local training.

---

**Happy Federated Learning!**

```
```
