from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


def merge_adapters_with_base_model(adapter_model_name: str, base_model_name: str, output_name: str, push_model_to_hub: bool):
    print(
        f"Merging Adapters weights {adapter_model_name} X {base_model_name} = {output_name}")
    peft_config = PeftConfig.from_pretrained(adapter_model_name)
    if peft_config.task_type == "SEQ_CLS":
        # The sequence classification task is used for the reward model in PPO
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1, torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, return_dict=True, torch_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the PEFT model
    model = PeftModel.from_pretrained(model, adapter_model_name)
    model.eval()

    model = model.merge_and_unload()

    model.save_pretrained(f"{output_name}")
    tokenizer.save_pretrained(f"{output_name}")
    if push_model_to_hub:
        model.push_to_hub(f"{output_name}")

#merge_adapters_with_base_model("/workspace/Finetuning-Pipeline/chat_test_base", "mistralai/Mistral-7B-v0.1", "SAGI-1/chat_test_base", True)
