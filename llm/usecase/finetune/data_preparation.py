import logging
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

# Set log format
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s", handlers=handlers_list)
logger = logging.getLogger(__name__)

def formatting_prompts_func(example):
    output_texts = []
    sys_prompt = "You are an expert trained on healthcare and biomedical reasoning."

    for i in range(len(example["instruction"])):
        text = (
            f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
            f"{example['instruction'][i]}\n[/INST]\n"
            f"### Response:\n{example['response'][i]} </s>"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return tokenizer, data_collator, formatting_prompts_func

def formatting(dataset):
    """Format dataset."""
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset

def reformat(dataset, llm_task):
    """Reformat datasets."""
    dataset = dataset.rename_column("output", "response")
    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])
    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset

def load_data(dataset_name: str, llm_task: str):
    """Load entire dataset without partitioning."""
    # 실제 데이터셋 로딩 필요
    dataset = load_dataset(dataset_name)

    # reformat 적용
    dataset = reformat(dataset, llm_task)
    return dataset

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
