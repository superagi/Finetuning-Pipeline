import os
from dataclasses import dataclass
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftConfig,
    PeftModel,
    AutoPeftModelForCausalLM
)
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer
from typing import Optional
from accelerate import Accelerator
from dataset_loader import DatasetLoader
import yaml
import locale

from merge_adapters import merge_adapters_with_base_model

accelerator = Accelerator()

# Device map
# DEVICE_MAP = {"": 0}​
DEVICE = "auto"
if torch.cuda.is_available():
    print("You have a GPU available! Setting `DEVICE=\"cuda\"`")
    DEVICE = accelerator.device


def clean_objects_and_empty_gpu_cache(arr: list, clear_cache: bool = True):
    """
    Use this function when you need to delete the objects, free their memory
    and also delete the cuda cache
    """
    for obj in arr:
        print(f"Deleting {obj}")
        del obj
    if clear_cache:
        torch.cuda.empty_cache()
        print("="*80)
        print("Cleared Cuda Cache")


locale.getpreferredencoding = lambda: "UTF-8"


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


config_path = "config.yaml"
config = load_config(config_path)

# Model Names
BASE_MODEL = config["BASE_MODEL"]
DATASET_NAME = config["DATASET_NAME"]
NEW_MODEL = config["NEW_MODEL"]

float_16_dtype = torch.float16
use_bf16 = config["use_bf16"]
use_4bit_bnb = config["use_4bit_bnb"]

compute_dtype = getattr(torch, "float16")

# Check GPU compatibility with bfloat16
# If the gpu is 'bf15' compatible, set the flag to `True`
if compute_dtype == torch.float16 and use_4bit_bnb:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("Changing floating point type to `torch.bfloat16`")
        float_16_dtype = torch.bfloat16
        use_bf16 = True
        print("=" * 80)
    else:
        print("Your GPU does not support bfloat16")
# Bits and Bytes configurations
# Used to quantize the model for memory saving
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit_bnb,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Supress fast_tokenizer warning
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

# Loading the model
if use_4bit_bnb:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        # device_map=DEVICE_MAP
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # device_map="auto"
        # device_map=DEVICE_MAP
    )
model.config.use_cache = False
model.config.pretraining_tp = 1

# View Model summary
# Will dictate the LoRA configuration:
# Specifically, which layers to fit the adapters to
print("=" * 80)
print("Model Summary")
print(model)
print("=" * 80)

# Some [optional] pre-processing which
# helps improve the stability of the training
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)

# Helper Function


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print("=" * 80)


LORA_TARGET_MODULES_LLAMA_2 = [
    "q_proj",
    "o_proj",
    "v_proj"
    "k_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=LORA_TARGET_MODULES_LLAMA_2,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

dataset_loader = DatasetLoader(DATASET_NAME)
train_dataset, test_dataset = dataset_loader.get_dataset()

# # Training arguments
OUTPUT_DIR = config["OUTPUT_DIR"]
LEARNING_RATE = config["LEARNING_RATE"]

NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
# effective backprop @ batch_size*grad_accum_steps
GRAD_ACCUMULATION_STEPS = config["GRAD_ACCUMULATION_STEPS"]
# speed down by ~20%, improves mem. efficiency
GRADIENT_CHECKPOINTING = config["GRADIENT_CHECKPOINTING"]

OPTIMIZER = config["OPTIMIZER"]
# OPTIMIZER = "AdamW"
# OPTIMIZER = "adamw_torch_fused" # use with pytorch compile
WEIGHT_DECAY = config["WEIGHT_DECAY"]
# examples include ["linear", "cosine", "constant"]
LR_SCHEDULER_TYPE = config["LR_SCHEDULER_TYPE"]
MAX_GRAD_NORM = config["MAX_GRAD_NORM"]  # clip the gradients after the value
# The lr takes 3% steps to reach stability
WARMUP_RATIO = config["WARMUP_RATIO"]

SAVE_STRATERGY = config["SAVE_STRATERGY"]
SAVE_STEPS = config["SAVE_STEPS"]
SAVE_TOTAL_LIMIT = config["SAVE_TOTAL_LIMIT"]
LOAD_BEST_MODEL_AT_END = config["LOAD_BEST_MODEL_AT_END"]

REPORT_TO = config["REPORT_TO"]
LOGGING_STEPS = config["LOGGING_STEPS"]
EVAL_STEPS = SAVE_STEPS

PACKING = config["PACKING"]
MAX_SEQ_LENGTH = config["MAX_SEQ_LENGTH"]

PUSH_TO_HUB = config["PUSH_TO_HUB"]
HF_MODEL_NAME = config["HF_MODEL_NAME"]

def calculate_steps():
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size / (BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Total number of steps: {total_steps}")


calculate_steps()

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,

    optim=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=not use_bf16,
    bf16=use_bf16,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,

    # torch_compile=False,
    group_by_length=False,

    save_strategy=SAVE_STRATERGY,
    save_steps=SAVE_STEPS,
    # save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,

    evaluation_strategy=SAVE_STRATERGY,
    eval_steps=EVAL_STEPS,

    dataloader_pin_memory=True,
    dataloader_num_workers=4,

    logging_steps=LOGGING_STEPS,
    report_to=REPORT_TO,
)

# Define the Supervised-Finetuning-Trainer from huggingface
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",

    args=training_arguments,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING,
)

# Train model from scratch
trainer.train()

# Save the model
trainer.model.save_pretrained(NEW_MODEL)

push_model_to_hub = config["PUSH_TO_HUB"]
if push_model_to_hub:
    merge_adapters_with_base_model(adapter_model_name=NEW_MODEL, base_model_name=BASE_MODEL, output_name=HF_MODEL_NAME)