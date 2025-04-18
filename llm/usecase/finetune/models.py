from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config.yaml")


def finetune_llm():
    def custom_train(model, train_dataset, tokenizer, formatting_prompts_func, data_collator):
        
        model.train()
        model.config.use_cache = False

        """Fine-Tune the LLM using SFTTrainer."""
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=cfg.finetune.learning_rate,
            per_device_train_batch_size=cfg.finetune.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.finetune.gradient_accumulation_steps,
            logging_steps=cfg.finetune.logging_steps,
            num_train_epochs=cfg.num_epochs,
            max_steps=cfg.finetune.max_steps,
            save_steps=cfg.finetune.save_steps,
            save_total_limit=cfg.finetune.save_total_limit,
            gradient_checkpointing=cfg.finetune.gradient_checkpointing,
            lr_scheduler_type=cfg.finetune.lr_scheduler_type,
            save_strategy="epoch",
            logging_dir="./logs",
            device_map="auto",
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=cfg.model.output_size,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator
        )

        trainer.train()

        from fedops.client.client_utils import get_parameters_for_llm
        return get_parameters_for_llm(model)

    return custom_train
