# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import yaml
from trl import DPOTrainer

def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
config_path = "config.yaml"
config = load_config(config_path)
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
        """
        # data parameters
    beta: Optional[float] = field(default=config["BETA"], metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default=config['MODEL_NAME'],
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=config["LEARNING_RATE"], metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default=config["LR_SCHEDULER_TYPE"], metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=config["WARMUP_STEPS"], metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=config["WEIGHT_DECAY"], metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default=config["OPTIMIZER_TYPE"], metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=config["PER_DEVICE_TRAIN_BATCH_SIZE"], metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=config["PER_DEVICE_EVAL_BATCH_SIZE"], metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=config["GRADIENT_ACCUMULATION_STEPS"],
        metadata={"help": "the number of gradient accumulation steps"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=config["GRADIENT_CHECKPOINTING"],
        metadata={"help": "whether to use gradient checkpointing"},
    )

    lora_alpha: Optional[float] = field(default=config["LORA_ALPHA"], metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=config["LORA_DROPOUT"], metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=config["LORA_R"], metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=config["MAX_PROMPT_LENGTH"], metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=config["MAX_LENGTH"], metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=config["MAX_STEPS"], metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=config["LOGGING_STEPS"], metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=config["SAVE_STEPS"], metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=config["EVAL_STEPS"], metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default=config["OUTPUT_DIR"], metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=config["LOG_FREQ"], metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=config["SANITY_CHECK"], metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=config["REPORT_TO"],
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=config["IGNORE_BIAS_BUFFERS"],
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )



def get_stack_exchange_paired(
    data_dir: str = config["DATASET_NAME"],
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    split = "train_prefs"
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        config["DATASET_NAME"],
        split=split,
        use_auth_token=config["HUGGINGFACE_AUTH_TOKEN"],
    )
    # print("Dataset: ", dataset)
    # print("Example of one row: ", dataset[0])
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [f"<s>[INST] {question} [/INST]" for question in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=config["LOAD_IN_4BIT"],
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=config["LOAD_IN_4BIT"],
    )
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(data_dir=config["DATASET_NAME"])
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_dir=config["DATASET_NAME"], split = "test_prefs")
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 4. initialize training arguments:

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=config["EVALUATION_STRATEGY"],
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        save_total_limit = config["SAVE_TOTAL_LIMIT"],
        bf16=config["USE_BF16"],
        remove_unused_columns=config["REMOVE_UNUSED_COLUMNS"],
        load_best_model_at_end=config["LOAD_BEST_MODEL_AT_END"],
        run_name="dpo_agi_1_2_multi_gpu_8_6000s_run_4",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    print("*"*80)
    print("TRAINING STARTED!!")
    print("*"*80)
    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)