import os
import numpy as np
import pandas as pd 

import hydra
from omegaconf import DictConfig
import datetime
from tqdm import tqdm
import wandb

import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

# get model and tokenizer
def get_model_and_tokenizer(
    args: DictConfig,         
) -> AutoModelForCausalLM:
    """Get model and tokenizer from config"""
    model = AutoModelForCausalLM.from_pretrained(**args.base_model, torch_dtype=torch.float16)
    tokenizer =  AutoTokenizer.from_pretrained(**args.tokenizer)
    return model, tokenizer

# tokenize
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result

# generate and tokenize prompt
def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt, tokenizer)
  

# run 
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:


   
    model, tokenizer = get_model_and_tokenizer(args.model)

    # breakpoint()

    # get data
    dataset = load_dataset("b-mc2/sql-create-context", split="train", cache_dir="/scr/jphilipp/printllama-hgx/data")
    subset_dataset = dataset.select(range(130))
    train_dataset = subset_dataset.select(range(100))
    eval_dataset = subset_dataset.select(range(100, 130))

    # add eos token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # tokenize
    tokenized_train_dataset = train_dataset.map(lambda data_point: generate_and_tokenize_prompt(data_point, tokenizer))
    tokenized_val_dataset = eval_dataset.map(lambda data_point: generate_and_tokenize_prompt(data_point, tokenizer))

    model.train() # put model back into training mode
    model = prepare_model_for_kbit_training(model)

    breakpoint()

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    resume_from_checkpoint = ""

    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            print(f"Restarting from {resume_from_checkpoint}")
            adapters_weights = torch.load(resume_from_checkpoint)
            set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

    wandb_project = "printllama"
    if len(wandb_project) > 0:
        os.environ["WANDB_API_KEY"] = "4d0657af83292fcbbcd01a9083fade1a1249ec3b"
        wandb.login()
        os.environ["WANDB_PROJECT"] = wandb_project

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    batch_size = 8
    per_device_train_batch_size = 4
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    output_dir = "/scr/jphilipp/printllama-hgx/finetuned_hf_models/codellama_7b_instruct_hf"

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1,
        max_steps=4,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=1,
        save_steps=1,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-test", # if use_wandb else None,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("compiling the model")
        model = torch.compile(model)

    trainer.train()

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=1 finetune.py model.codellama_7b
# python finetune.py --config-name=model/codellama_7b
