```markdown
# Multimodal Federated Hateful Memes Classification (FedOps Silo)

This repository demonstrates a full end-to-end **multimodal** (text + image) federated learning (FL) setup using FedOps Silo and Flower, applied to the [Hateful Memes](https://huggingface.co/neuralcatcher/hateful_memes) dataset. Each client trains a fusion model on its local (non-IID) split of the memes, and a centralized FL server aggregates updates across all clients. We also show how to customize both the **model‐level fusion** (inside `models.py`) and the **server‐side aggregation** (custom Flower Strategy).

---

## 🚀 Features

- **Multimodal Fusion (Early, Intermediate, Late)**  
  We demonstrate multiple fusion strategies for combining text and image modalities:
  1. **Early Fusion**: Concatenate raw or preprocessed text embeddings and image embeddings before any deeper processing.  
  2. **Intermediate Fusion**: Use cross‐attention or other attention mechanisms to let one modality guide the other (e.g. image attends over text features or vice versa).  
  3. **Late Fusion**: Train separate unimodal classifiers for text and image, then combine their output logits or probabilities in a final decision layer.  

  The default `HatefulMemesFusionModel` uses a simple fusion layer (concatenation of BERT pooled output and CNN image features). You can also implement a cross-attention variant (Intermediate Fusion) under `multimodal/FusionModel/custom_fusion.py`.

- **Federated Learning Stack**  
  Uses [FedOps Silo](https://github.com/fedops-fedmm) and [Flower](https://flower.dev) under the hood:
  1. **Clients** each run a local PyTorch training loop (text + image).
  2. **Server** uses a Flower Strategy (e.g. FedAvg or a custom aggregator) to merge client weights each round.

- **Non-IID Data Splits**  
  Each client receives a stratified, non-IID subset of the Hateful Memes dataset (both text & images). A small “hold-out” validation set remains with the server for global evaluation.

- **GPU Acceleration**  
  Clients detect and utilize any available GPU for faster model training (CUDA + cuDNN).

- **Custom Aggregation Strategies**  
  Easily add your own FL Strategy (e.g. modality‐aware weight fusion) under `multimodal/Aggregation/…` and point to it in `conf/config.yaml`.

- **Hydra Configuration**  
  All hyperparameters, model settings, and FL server/strategy parameters live in a single `conf/config.yaml` file that works seamlessly for both server & client.

---

## 📂 Folder Structure

```

Hateful\_memes\_classification/
├── clienta\_main.py          # “Client A” entry point (GPU setup + FL client)
├── clientb\_main.py          # “Client B” entry point (similar to clienta\_main.py)
├── client\_manager\_main.py   # Flower client manager (registers client with central coordinator)
├── server\_main.py           # FL server entry point (starts Flower server)
├── models.py                # Defines `HatefulMemesFusionModel` (text+image fusion)
├── data\_preparation.py      # Loads Hateful Memes dataset, returns PyTorch DataLoaders
├── multimodal/
│   ├── FusionModel/
│   │   └── custom\_fusion.py # (optional) alternative fusion architectures (e.g. cross‐attention)
│   └── Aggregation/
│       └── my\_custom\_strategy.py   # Example custom Flower Strategy subclass
├── conf/
│   └── config.yaml           # Hydra config (model, FL server, strategy, hyperparams)
├── requirements.txt          # Python dependencies
└── README.md                 # ← You are reading this file

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
````

Make sure your CUDA drivers and PyTorch CUDA build are correctly installed if you intend to use a GPU.

---

### 2. Configuration (conf/config.yaml)

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
  validation_split: 0.2  # 20% of full data held out at server

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
    # Point to either FedAvg or your own custom strategy:
    _target_: multimodal.Aggregation.my_custom_strategy.MyCustomStrategy
    alpha: 0.7         # Example hyperparameter for custom Strategy
    fraction_fit: 0.00001
    fraction_evaluate: 0.000001
    min_fit_clients: ${clients_per_round}
    min_available_clients: ${clients_per_round}
    min_evaluate_clients: ${clients_per_round}
```

*Key points:*

* `_target_` under `model:` must point to the PyTorch fusion model class in `models.py` (or a custom fusion file under `multimodal/FusionModel/…`).
* `_target_` under `server.strategy:` can be Flower’s built-in `FedAvg` or your own class (e.g. `multimodal.Aggregation.my_custom_strategy.MyCustomStrategy`).

---

### 3. Non-IID Data Splitting

Each client loads its own **local** (non-IID) partition of the Hateful Memes dataset. By default, `data_preparation.py` uses the **entire** dataset in each client, but you can customize:

```python
# data_preparation.py

def load_partition(batch_size=32, split="train", client_id=None):
    full_dataset = load_dataset("neuralcatcher/hateful_memes", split=split)
    if client_id is not None:
        # Example: stratified 50/50 split by “label”
        indices = stratified_indices_for_client(full_dataset["label"], client_id, num_clients=2)
        local_subset = full_dataset.select(indices)
    else:
        local_subset = full_dataset

    # … tokenize, transform images, return DataLoader(local_subset) …
```

If you want a gold-standard non-IID (e.g. Dirichlet allocation or label partition), implement `stratified_indices_for_client(...)` accordingly in `data_preparation.py` and pass `client_id=0` or `1` when each client calls `load_partition(...)`.

---

### 4. Running Clients & Server

You’ll start:

1. **Server** (on a machine with Kubernetes or just a local shell):

   ```bash
   # In one terminal:
   source venv/bin/activate
   python server_main.py --config-name=config
   ```

   This spins up the Flower server on port `8787` by default (or as configured in `config.yaml`).

2. **Client Manager** (on each client machine):

   ```bash
   # In separate terminal on Client A:
   source venv/bin/activate
   python client_manager_main.py
   ```

   This FastAPI client manager polls the server for “start training” signals.

3. **Client Training Process** (on each client machine, in a different terminal):

   ```bash
   # On Client A:
   source venv/bin/activate
   python clienta_main.py --config-name=config
   ```

   * Client detects GPU (if available) and prints: `🔌 Using device: cuda` or `cpu`.
   * It registers with the manager, waits for a `/start` signal from `client_manager_main.py`, then spins up Flower’s NumPyClient under the hood and starts local training.

Repeat similarly on **Client B** machine:

```bash
# On Client B:
python clientb_main.py --config-name=config
```

Once both clients are live, the server’s strategy (e.g. FedAvg or your custom `MyCustomStrategy`) will accept their weight updates each round, aggregate them, and broadcast back the new global model.

---

## 🖥️ GPU Acceleration (Client Side)

Each client’s `clienta_main.py` / `clientb_main.py` includes a GPU check:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
```

The fusion model is then moved onto that device:

```python
model = instantiate(cfg.model)
model = model.to(device)
```

Inside `models.py`, both train and test loops send tensors to `device` with `batch["…"].to(device, non_blocking=True)`. If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES=0,1` when launching the client.

---

## ⚙️ Custom Aggregation Strategy

To add your own FL aggregator:

1. **Create a new file** under `multimodal/Aggregation/my_custom_strategy.py`:

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
           Example: Compute a weighted average of client weights
           where each client’s weight = alpha * (validation_accuracy)
           + (1 - alpha) * (num_samples / total_samples).
           """
           weighted_results = []
           total_samples = sum(res.metrics["num_examples"] for _, res in results)

           for client, fit_res in results:
               local_samples = fit_res.metrics["num_examples"]
               local_acc = fit_res.metrics.get("accuracy", 0.0)
               w_size   = local_samples / total_samples
               # Combine performance and data size:
               weight_factor = self.alpha * local_acc + (1.0 - self.alpha) * w_size
               weighted_results.append((client, fit_res, weight_factor))

           # Call FedAvg with our custom weight_factors:
           return super().aggregate_fit(rnd, [
               (client, fit_res, weight_factor) for client, fit_res, weight_factor in weighted_results
           ], failures)
   ```

2. **Update `conf/config.yaml`** to point to this new class:

   ```yaml
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

When the FL server starts, Hydra will resolve `_target_` to your custom strategy class, instantiate it with `alpha=0.7`, and run it.

---

## 🔄 Custom Model Fusion

We support multiple fusion paradigms in `models.py` (late fusion by default) and custom fusion files under `multimodal/FusionModel/…` for early or intermediate approaches:

1. **Early Fusion**:

   * Simply concatenate text embeddings (e.g. pooled BERT output) and image embeddings (e.g. CNN features) right after each encoder.
   * Example (in `models.py` default):

     ```python
     text_features = self.bert(input_ids, attention_mask).pooler_output  # (batch, 768)
     image_feat    = self.image_encoder(image)                           # (batch, 128)
     fused         = torch.cat((text_features, image_feat), dim=1)       # (batch, 768+128)
     out           = self.classifier(F.relu(self.fusion_fc(fused)))      # final logits
     ```

2. **Intermediate Fusion (Cross‐Attention)**:

   * Use one modality as “query” and the other as “key/value” in a MultiheadAttention block.
   * For example, let the image features query the text features:

     ```python
     # In multimodal/FusionModel/custom_fusion.py:
     text_feat = self.bert(...).pooler_output.unsqueeze(1)         # (batch, 1, 768)
     img_feat  = self.image_encoder(image).view(batch, -1).unsqueeze(0)  # (1, batch, 128)
     attn_out, _ = self.cross_attention(
         query=img_feat, 
         key=text_feat.transpose(0,1), 
         value=text_feat.transpose(0,1)
     )
     attn_out = attn_out.squeeze(0)  # (batch, 768)
     fused    = F.relu(self.fusion_fc(torch.cat([attn_out, img_feat.squeeze(0)], dim=1)))
     out      = self.classifier(fused)
     ```

3. **Late Fusion**:

   * Train separate text‐only and image‐only classifiers, then combine their logits in a final decision layer or weighted sum.
   * Example (create two heads in `models.py`):

     ```python
     text_logits  = self.text_head(text_features)   # (batch, 2)
     image_logits = self.image_head(image_features) # (batch, 2)
     final_logits = self.late_fusion_layer(text_logits, image_logits) 
     # Could be weighted average: alpha * text_logits + (1-alpha) * image_logits
     ```

To try a new fusion style, place your class under `multimodal/FusionModel/…` and update `_target_` in `conf/config.yaml`.

---

## 📖 Usage Workflow

1. **Prepare K8s (Optional)**
   If you want to run on Kubernetes, build Docker images for:

   * FL server (includes `server_main.py` + dependencies)
   * Each client (includes `client_manager_main.py` + `clienta_main.py` or `clientb_main.py`)

   Ensure nodes have GPUs (set appropriate resource requests/limits). If pods get evicted due to `out-of-memory` or `out-of-shared-memory`, you may need to increase `spec.containers.resources.limits.memory` or `spec.containers.securityContext.privileged` for `shmSize`.

2. **Launch FL Server**

   ```bash
   python server_main.py --config-name=config
   ```

   * The server loads its **hold-out validation set** (20% of full data, stratified) via `data_preparation.gl_model_torch_validation`.
   * Starts Flower on `0.0.0.0:8787` (default) and waits for clients.

3. **Launch Client Manager**

   ```bash
   python client_manager_main.py
   ```

   * The FastAPI manager polls the server at `/FLSe/info` every few seconds.
   * When the server signals “start training,” the manager sends a `/start` request to the local FL client.

4. **Launch FL Client**

   ```bash
   python clienta_main.py --config-name=config
   # On another machine or different terminal:
   python clientb_main.py --config-name=config
   ```

   * Each client prints:

     ```
     🔌 Using device: cuda
     ```

     if CUDA is available, else `cpu`.
   * Clients load their local (non-IID) subset of Hateful Memes via `data_preparation.load_partition(...)`.
   * Each client manager will signal “start training” once both are registered online.
   * The FL client uses Flower’s `start_numpy_client` under the hood to communicate with the server.

5. **Monitoring & Logs**

   * Server logs round‐by‐round metrics: initial validation loss/accuracy, then after each aggregation.
   * Clients log local train/validation metrics each round.
   * If a client times out or loses connection, you’ll see a “ping timeout” or `StatusCode.UNAVAILABLE`; the manager may automatically mark it offline and notify the server.

6. **Hold-Out Validation on Server**

   * The server’s `FLServer` uses `gl_val_loader` (20% hold-out) to compute global model performance each round, independent of client validation.
   * This ensures the server never “sees” any training splits from clients, even indirectly.

---

## 📚 Extending & Publishing

1. **Add New Aggregation Strategies**

   * Put each new strategy class under `multimodal/Aggregation/…`.
   * Update `_target_` in `conf/config.yaml`.
   * Run `python server_main.py` to test locally.

2. **Convert to a Standalone Python Package**
   Once your strategy(s) are rock-solid:

   * Create a `setup.py` or `pyproject.toml` at project root.
   * Under `packages=[ "multimodal.Aggregation", … ]` list your new module.
   * Ensure dependencies are in `install_requires=[ … ]`.
   * Then:

     ```bash
     python3 -m build
     twine upload dist/*
     ```
   * Users can then `pip install your-package` and point `_target_` to `your_package.my_custom_strategy.MyCustomStrategy`.

3. **Contribute Back to FedOps / Flower**
   If your aggregator outperforms existing multimodal strategies, consider opening a PR under the official FedOps repo or publishing a tutorial on Flower’s website.

---

## ✨ Acknowledgments

* **FedOps Silo** – Federated learning boilerplate for PyTorch & Flower.
* **Flower** – A friendly, lightweight federated learning framework.
* **Hateful Memes** – A multimodal dataset (text + images) for offensive content classification.
* **HuggingFace transformers** – BERT model for text encoding.
* **PyTorch** – Deep learning library powering local training.

---

**Happy Federated Learning!**

Feel free to open issues or pull requests in this repo for bug fixes, improvements, or new custom strategies.

```
```


## ✅ Key Hyperparameters

All of these live in `conf/config.yaml`. Change and re-run:

```yaml
random_seed: 42
lr: 0.0001
model_type: 'Pytorch'

# Model architecture:
model:
  _target_: models.HatefulMemesFusionModel
  text_hidden_dim: 768
  image_output_dim: 128
  fusion_output_dim: 256
  output_size: 2

dataset:
  name: 'hateful_memes'
  validation_split: 0.2

# Federated setup:
task_id: 'hatetaskthree'
num_epochs: 1
batch_size: 32
num_rounds: 2
clients_per_round: 2

# Flower Strategy:
server:
  strategy:
    _target_: flwr.server.strategy.FedAvg
    fraction_fit: 1.0      # e.g. 1.0 for all selected clients
    fraction_evaluate: 0.5
    min_fit_clients: ${clients_per_round}
    min_available_clients: ${clients_per_round}
    min_evaluate_clients: ${clients_per_round}
```

* **`num_epochs`**: how many local epochs each client runs per round.
* **`batch_size`**: local mini-batch size.
* **`num_rounds`**: total FL rounds.
* **`clients_per_round`**: how many clients to sample per FL round.
* **`fraction_fit`** & **`fraction_evaluate`**: these can be 1.0 (use all), or a fraction if you have > 2 clients.

---

## 🔍 Troubleshooting

* **Missing GPU usage**

  * Ensure `torch.cuda.is_available()` returns `True`.
  * In `clienta_main.py`, you should see `🔌 Using device: cuda:0`.
  * If it prints `cpu`, check that you have a CUDA-compatible GPU and proper drivers.

* **“⚠️ Missing image or URL” logs**

  * Some Hateful Memes entries may lack a valid image URL. The dataset loader prints a warning and substitutes a zero-tensor. This does not break training; accuracy may degrade slightly.

* **Flower client “UNAVAILABLE” / “ping timeout”**

  * Check that the server’s gRPC port (40025 by default) is reachable from each client node (firewall, Kubernetes Service config).
  * Verify `server_main.py` actually started without errors.

* **Kubernetes eviction / OOM**

  * The image-loading in `data_preparation.py` can consume shared memory. Increase `emptyDir.medium: "Memory"` or raise `–shm-size` if using Docker.
  * In Kubernetes, ensure your Pod spec requests enough RAM (e.g., `resources.requests.memory: "8Gi"`).

---

## 📚 References

* **Flower**: “Flower: A Friendly Federated Learning Research Framework” ([https://flower.dev/](https://flower.dev/))
* **FedOps**: “FedOps: Federated Operations Made Easy” ([https://github.com/FedOps/FedOps](https://github.com/FedOps/FedOps))
* **Hateful Memes**: Kiela et al., “The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes” (NeurIPS 2020)
* **Hugging Face Datasets**: “neuralcatcher/hateful\_memes” on Hugging Face.

---

## 🏷 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* Built with ❤️ using PyTorch, Hugging Face Transformers, Flower, and FedOps.
* Inspired by “NeuralCatcher/Hateful Memes” on Hugging Face.
