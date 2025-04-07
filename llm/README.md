# FedOps LLM Fine-Tune

This guide provides step-by-step instructions on how to implement FedOps LLM Fine-Tune, a federated learning lifecycle management operations framework.

This use case will work just fine without modifying anything.

## Baseline

```
- Baseline
    - generate_paramshape.py
    - client_main.py
    - client_mananger_main.py
    - server_main.py
    - models.py
    - data_preparation.py
    - requirements.py (for server)
    - conf
        - config.yaml
```

## Step

1. **Start by cloning the FedOps**

```
git clone https://github.com/gachon-CCLab/FedOps.git \
&& mv FedOps/llm/usecase/finetune . \
&& rm -rf FedOps
```

2. **Customize the FedOps Baseline code.**

* Customize the FedOps llm fine-tune code to align with your FL task.

  * config.yaml
    * model.name: Select from the models of Hugging Face.
      * The current model is `DeepSeek-R1-Distill-Qwen-1.5B`, which requires about 27 GB of GPU memory.
    * dataset.name: Select from the datasets of HuggingFace.
      * The current dataset is `medical_meadow_medical_flashcards` and is set for medical fine-tuning.
    * task_id: Your taskname that you register in FedOps Website.
    * finetune: Modify LoRA hyperparameter settings.
    * num_epochs: Number of local learnings per round for a client.
    * num_rounds: Number of rounds.
    * clients_per_round: Number of clients participating per round.
  * If you want to change the dataset.
    * You need to modify the files below.
      * data_preparation.py
      * client_main.py

3. **Run** `generate_paramshape.py`

- This will generate `parameter_shape.json`, which is ready for global model aggregation.

4. **Make it your own GitHub repository**

5. **Login to the FedOps Website**

* Create a Task.
  * Title: It must be the same as the task_id specified in config.
  * Client Type: Silo
  * Description: Your own task description
  * Server Git Repository Address: Repository address created in step 4

6. **Start federated learning**

- Client
  - start `client_main.py` & `client_manager_main.py`
- Task window of FedOps Website
  - Select your clients who are online and press `FL START`.

Once training is complete, you can download the global model (LoRA Adapter) from the Global Model window.

If you want to try this, try the code below.

```
import numpy as np
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import OrderedDict
from flwr.common.typing import NDArrays
import torch

# .npz load
loaded = np.load("Your own path")

print(loaded.files)  # → ['arr_0', 'arr_1', 'arr_2', ...]

parameters = [loaded[k] for k in loaded.files]

def set_parameters_for_llm(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def load_model(model_name: str, quantization: int, gradient_checkpointing: bool, peft_config):
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

        if gradient_checkpointing:
            model.config.use_cache = False

        return get_peft_model(model, peft_config)

quantization = 4
gradient_checkpoining = True
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.075,
    task_type="CAUSAL_LM",
)

model = load_model(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization=quantization,
    gradient_checkpointing=gradient_checkpoining,
    peft_config=peft_config,
)

set_parameters_for_llm(model, parameters)


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

input_text = "hello"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"🧠 Generated:\n{generated_text}")
```
