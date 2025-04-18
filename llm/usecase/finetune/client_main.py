import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models

from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf

from fedops.client.client_utils import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    
    
@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # set log format
    handlers_list = [logging.StreamHandler()]
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

    logger = logging.getLogger(__name__)
    
    # Set random_seed
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    print(OmegaConf.to_yaml(cfg))
    
    """
    Load Tokenizer, Data Collator, Prompt Formatting Function
    """
    tokenizer, data_collator, formatting_prompts_func = data_preparation.get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    logger.info(f"Loaded tokenizer and data collator for model {cfg.model.name}")

    """
    Load Dataset
    """
    dataset = data_preparation.load_data(dataset_name=cfg.dataset.name, llm_task=cfg.dataset.llm_task)
    # formatted_dataset = dataset.map(prompt_formatting, remove_columns=["instruction", "response"])  # 프롬프트 변환

    # Preprocess function
    def preprocess_function(examples):
        texts = formatting_prompts_func(examples)
        model_inputs = tokenizer(
            texts, padding="max_length", truncation=True, max_length=512
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    # Tokenize datasets
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_dataset["train"]

    logger.info("Dataset formatted Complete")

    """
    Model Load
    """
    
    quantization = 4
    gradient_checkpoining = True

    peft_config = LoraConfig(
        r=cfg.finetune.lora_r,
        lora_alpha=cfg.finetune.lora_alpha,
        lora_dropout=cfg.finetune.lora_dropout,
        task_type="CAUSAL_LM",
    )

    model = load_model(
        model_name=cfg.model.name, 
        quantization=quantization,
        gradient_checkpointing=gradient_checkpoining,
        peft_config=peft_config,
    )


    """
    Fine-Tune the Model using finetune_llm from models.py
    """
    finetune_llm = models.finetune_llm()

    
    """
    Register model and dataset for Federated Learning
    """
    registration = {
        "finetune_llm": finetune_llm,
        "trainset" : train_dataset,
        "model" : model,
        "model_name": cfg.model.name,
        "tokenizer" : tokenizer,
        "formatting_prompts_func" : formatting_prompts_func,
        "data_collator" : data_collator,
    }
    
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
    

