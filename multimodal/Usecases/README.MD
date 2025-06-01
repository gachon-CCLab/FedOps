```markdown
# FedOps • Hateful Memes Federated Learning

This repository demonstrates a PyTorch‐based federated learning (FL) setup using [FedOps](https://github.com/FedOps/FedOps) and [Flower](https://flower.dev/) on the “NeuralCatcher/Hateful Memes” multimodal classification dataset. Each client trains a simple BERT + CNN fusion model on local image + text splits, and a central FL server (FedOps/Flower) aggregates updates via FedAvg. You can easily switch to non‐IID data splits, customize aggregation strategies, or plug in your own FL algorithm.

---

## 🔎 Project Overview

- **Dataset**:  
  [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes)  
  Combines meme images with overlaid text and binary labels (hateful vs. non‐hateful).

- **Model**:  
  - **Text branch**: pre‐trained BERT (pooler output of size 768)  
  - **Image branch**: simple 2-layer CNN → adaptive pooling → fully connected to 128  
  - **Fusion**: concatenate [768 ⊕ 128] → 256-dim fusion → 2‐class classifier  

- **Federated Learning**:  
  - Flower’s FedAvg strategy (configurable via Hydra).  
  - “Hold‐out” 10% validation on the server side for global evaluation.  
  - Clients each load (IID by default) full training data—but you can replace with non‐IID splits.  
  - A lightweight FastAPI “Client Manager” on each node triggers `/start`/`/stop` for local training.

---

## 📂 Repository Structure

```

Hateful\_memes\_classification/
├── conf/
│   └── config.yaml         # Hydra config: hyperparams + FL strategy
│
├── data\_preparation.py     # PyTorch Dataset + DataLoader logic for Hateful Memes
├── models.py               # Fusion model definition + train/test helpers
│
├── server\_main.py          # Hydra → FedOps FLServer entrypoint
│
├── clienta\_main.py         # Hydra → FedOps FLClient entrypoint (actual local training)
├── client\_manager\_main.py  # FastAPI “manager” that polls server and triggers client
│
└── requirements.txt        # All Python dependencies

```

- **conf/config.yaml**  
  All shared hyperparameters (batch size, learning rate, number of rounds, model architecture, FL strategy, etc.). Every Python script uses Hydra to load the same config.

- **data_preparation.py**  
  - `HatefulMemesDataset(split, max_length)` → loads Hateful Memes via HuggingFace Datasets, tokenizes text with BERT, downloads + preprocesses image.  
  - `load_partition(batch_size)` → returns `(train_loader, val_loader, test_loader)` for a single client.  
  - `gl_model_torch_validation(batch_size)` → server’s 10% hold‐out validation loader (no client sees this during training).

- **models.py**  
  - `HatefulMemesFusionModel` → multi‐modal fusion network.  
  - `train_torch()` → returns a function `train(model, loader, epochs, cfg)` that moves everything to GPU/CPU, performs local updates, and returns the updated model.  
  - `test_torch()` → returns a function `test(model, loader, cfg)` that evaluates on validation/test data and returns `(loss, accuracy, metrics_dict)`.

- **server_main.py**  
  - Reads `conf/config.yaml` via Hydra.  
  - Instantiates initial global model (`instantiate(cfg.model)`).  
  - Builds a single global validation loader via `gl_model_torch_validation`.  
  - Wraps `test_torch()` into `test_wrapper(model, loader, cfg)` for server‐side evaluation.  
  - Creates and runs `FLServer(cfg, model, model_name, model_type, gl_val_loader, test_wrapper)`, which spins up Flower’s FL server loop.

- **clienta_main.py**  
  - Reads `conf/config.yaml` via Hydra.  
  - Selects GPU if available, prints device.  
  - Sets random seeds.  
  - Loads local `(train_loader, val_loader, test_loader) = load_partition(cfg.batch_size)`.  
  - Instantiates fusion model → `.to(device)`.  
  - Downloads any local checkpoint if exists (`client_utils.download_local_model`).  
  - Prepares `train_torch()` and `test_torch()`.  
  - Registers everything in `registration = {train_loader, val_loader, test_loader, model, model_name, train_torch, test_torch}`.  
  - Launches `fl_client = FLClientTask(cfg, registration)` and calls `fl_client.start()` to connect to Flower server.

- **client_manager_main.py**  
  - A FastAPI app that periodically:  
    1. Polls server’s “/info” endpoint (via `requests.get(http://<serverST>/FLSe/info/<task_id>/<mac>)`).  
    2. Once server signals “ready” for the next round, calls `POST http://localhost:8003/start {server_ip, client_mac}` on the local client.  
    3. Implements `/trainFin` and `/trainFail` endpoints for the client to report status.  
  - Ensures each client does not begin local training until the FL server authorizes it.

- **requirements.txt**  
```

fedops
torch
torchvision
scikit-learn
pandas
numpy
tqdm

transformers
datasets

Pillow
requests

uvicorn
fastapi
pydantic
hydra-core
omegaconf

````
Install with:
```bash
pip install -r requirements.txt
````

---

## ⚙️ Installation / Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-org>/Hateful_memes_classification.git
   cd Hateful_memes_classification
   ```

2. **Create a Python 3.9+ virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Verify GPU availability**

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 🚀 Running the FL Server

On a dedicated “server” machine (or Kubernetes pod), run:

```bash
cd Hateful_memes_classification
python server_main.py
```

Hydra will load `conf/config.yaml` and start Flower’s FL server. You should see logs like:

```
ℹ️  Building global model: HatefulMemesFusionModel
ℹ️  Loading validation split for global evaluation (10% hold‐out)
ℹ️  Flower server running: … (port 40025)
ℹ️  [ROUND 0] initial (loss, metrics): 0.26, {‘accuracy’: 0.58}
```

By default, Flower’s gRPC port is `40025` (you can override via environment variables or Hydra if needed).

---

## 🚀 Running a Single Client (Stand‐alone)

If you want to test locally with just one client (no manager), you can do:

```bash
cd Hateful_memes_classification
# In one terminal: start server_main.py
python server_main.py

# In a second terminal: start client_main.py (renamed to clienta_main.py here)
python clienta_main.py
```

You will see:

```
🔌 Using device: cuda:0
🔧 Step 1: Setting random seeds
… (data loading logs)
🧠 Step 4: Instantiating fusion model
🚀 Step 8: Launching FL client task…
[flwr][INFO] Connecting to server at ccl.gachon.ac.kr:40025
…
```

That single client will join Round 1 immediately. Since `clients_per_round=2` in `config.yaml`, you might need at least two clients to fully kick off FedAvg. To simulate two local clients on one machine, run `python clienta_main.py` in two separate shells (they’ll both contact `localhost:40025` by default). Each will load the full dataset (IID) and train locally for `num_epochs=1` before uploading.

---

## 🚀 Running with Client Manager (Kubernetes Style)

In production/Kubernetes, each client runs two processes:

1. **Client Manager** (FastAPI, port 8004)

   ```
   python client_manager_main.py
   ```
2. **FL Client** (Hydra + Flower, port 8003)

   ```
   python clienta_main.py
   ```

The manager polls the server’s `/info` endpoint every few seconds. When Flower server is ready for Round n, the manager calls:

```
POST http://localhost:8003/start
  { "server_ip": "<server‐hostname>", "client_mac": "<device_mac_address>" }
```

The client then begins local training, and upon completion hits `/trainFin` or `/trainFail` on the manager. This two‐process pattern is useful in Kubernetes so that your container doesn’t block solely on Flower’s long‐running local loop.

---

## 🔀 Non‐IID Data Splits

By default, each client calls:

```python
train_loader, val_loader, test_loader = load_partition(batch_size=cfg.batch_size)
```

which uses the **entire** Hateful Memes training split. To create a “gold‐standard” non‐IID partition:

1. **Modify** `data_preparation.py` to support splitting `self.dataset` by label or text cluster. For example:

   ```python
   def load_non_iid_partition(client_id: str, batch_size: int):
       full = load_dataset("neuralcatcher/hateful_memes", split="train")
       # e.g. stratify by label: client “A” gets all negative-label memes, client “B” gets positive-label
       # Or cluster by text features → partition accordingly
       client_subset = full.filter(lambda x: x["label"] == desired_label_for_client(client_id))
       train_data = HatefulMemesDatasetFromHF(client_subset, split="train‐pseudo", max_length=128)
       # … same for validation/test or let server hold them out
       train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, …
       return train_loader
   ```
2. In each client’s entrypoint (`clienta_main.py` vs. `clientb_main.py`), call:

   ```python
   train_loader = load_non_iid_partition(client_id="A", batch_size=cfg.batch_size)
   ```

   rather than `load_partition(...)`.
3. Keep the server’s `gl_model_torch_validation(...)` unchanged: it will load 10% hold‐out from the full split.

This guarantees that no client sees the entire distribution and each has its own skewed subset.

---

## 📦 Custom Aggregation Strategy

If you want to implement a novel FL aggregation (e.g. “Modality‐Aware FedAvg”), you can:

1. **Create a new Python file** under `fedops/server/strategies/`, for example:

   ```
   fedops/server/strategies/multimodal_aware.py
   ```

   and define your custom strategy class inheriting from `flwr.server.strategy.Strategy`, overriding `aggregate_fit` or other hooks.

2. **Modify** `conf/config.yaml`:

   ```yaml
   server:
     strategy:
       _target_: fedops.server.strategies.multimodal_aware.MyCustomStrategy
       # … any extra hyperparameters for your strategy
   ```

3. In `server_main.py`, because the config’s `server.strategy._target_` now points to your custom class, FedOps will automatically instantiate it instead of `FedAvg`.

Later, you can package `fedops/server/strategies/multimodal_aware.py` (and any necessary helper modules) into a PyPI package so others can install via `pip install your‐package`. For now, keep it inside `fedops/server/strategies/` and reference it in `config.yaml`.

---

## 📝 Example Commands

1. **Install dependencies** (on each node/pod):

   ```bash
   git clone https://github.com/<yourorg>/Hateful_memes_classification.git
   cd Hateful_memes_classification
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Start FL Server** (port 40025 by default):

   ```bash
   python server_main.py
   ```

3. **Start Client Manager** (port 8004):

   ```bash
   python client_manager_main.py
   ```

4. **Start FL Client** (port 8003):

   ```bash
   python clienta_main.py
   ```

   * For two clients on one machine, simply open two terminals and run both the manager + client pair twice (each gets its own device MAC, so Flower sees 2 distinct clients).

5. **Monitor Logs**

   * Server logs: round‐by‐round validation accuracy.
   * Client logs: “train\_performance → {train\_loss, train\_accuracy, val\_loss, val\_accuracy}.”
   * Manager logs: health checks, start‐training triggers, and final success/fail messages.

---

## ✅ Key Hyperparameters

All of these live in `conf/config.yaml`. Change and re‐run:

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
* **`batch_size`**: local mini‐batch size.
* **`num_rounds`**: total FL rounds.
* **`clients_per_round`**: how many clients to sample per FL round.
* **`fraction_fit`** & **`fraction_evaluate`**: these can be 1.0 (use all), or a fraction if you have >2 clients.

---

## 🔍 Troubleshooting

* **Missing GPU usage**:

  * Ensure `torch.cuda.is_available()` returns `True`.
  * In `clienta_main.py`, you should see `🔌 Using device: cuda:0`.
  * If it prints `cpu`, check that you have a CUDA‐compatible GPU and proper drivers.

* **“⚠️ Missing image or URL” logs**:

  * Some Hateful Memes entries may lack a valid image URL. The dataset loader prints a warning and substitutes a zero‐tensor. This does not break training; accuracy may degrade slightly.

* **Flower client “UNAVAILABLE” / “ping timeout”**:

  * Check that server’s gRPC port (40025 by default) is reachable from each client node (firewall, Kubernetes Service config).
  * Verify `server_main.py` actually started without errors.

* **Kubernetes eviction / OOM**:

  * The image‐loading in `data_preparation.py` can consume shared memory. Increase `emptyDir.medium: “Memory”` or raise `–shm-size` if using Docker.
  * In K8s, ensure your Pod spec requests enough RAM (e.g., `resources.requests.memory: “8Gi”`).

---

## 📚 Citations & References

* **Flower**: “Flower: A Friendly Federated Learning Research Framework” ([https://flower.dev/](https://flower.dev/))
* **FedOps**: “FedOps: Federated Operations Made Easy” ([https://github.com/FedOps/FedOps](https://github.com/FedOps/FedOps))
* **Hateful Memes**: Kiela et al., “The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes” (NeurIPS 2020)
* **Hugging Face Datasets**: “neuralcatcher/hateful\_memes” on HF.

---

## 🏷 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* Built with ❤️ using PyTorch, HuggingFace Transformers, Flower, and FedOps.
* Inspired by “NeuralCatcher/Hateful Memes” on Hugging Face.

```
```
