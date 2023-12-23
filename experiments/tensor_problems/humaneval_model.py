import hydra
from omegaconf import DictConfig
import argparse
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
import random
import re
import time
import os
from human_eval.data import write_jsonl, read_problems

from printllama.helpers import extract_code


# import models
from printllama.models.codellama_meta.generation import Llama
from printllama.models.huggingface.hf_inference_model import HFInferenceModel
from printllama.models.openai.azure import AsyncAzureChatLLM
from printllama.models.openai.gpt4 import GPT4Agent

# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Running inference model...")

    seed = 1

    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model.model_type.lower()
    is_hf = "hf" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()


    # GET DATA
    start = time.time()
    with open(args.data.data_path, "r") as f:
        data = json.load(f)
    print(f"==== Data loaded in {time.time() - start} seconds ====")

    problems = read_problems()
    num_samples_per_task = 100
    # BUILD MODEL AND RUN INFERENCE
    if not args.model.run.verbose:
        if is_meta:
            start = time.time()
            model = Llama.build(**args.model.model_config)
            print(f"==== Model built in {time.time() - start} seconds ====")
            batched_prompts = [data] * args.model.run.batch_size
            print(f"Began batch prompting...")
            start = time.time()
            completions = model.chat_completion(
                batched_prompts,
                **args.model.run.completion_config,
            )
            print(f"==== Completions generated in {time.time() - start} seconds ====")
        elif is_hf: 
            model = HFInferenceModel(**args.model.model_config)
            if 'mistral' in args.model.name.lower():
                print(f"Began batch prompting...")
                samples = [dict(task_id=task_id, completion=completion) for completion in model.batch_prompt([problems[task_id]["prompt"]] * args.model.run.batch_size, **args.model.run.completion_config) for task_id in problems]
                print(f"==== Completions generated in {time.time() - start} seconds ====")
            else:
                print(f"Began batch prompting...")
                samples = [dict(task_id=task_id, completion=completion) for completion in model.batch_prompt([problems[task_id]["prompt"]] * args.model.run.batch_size, **args.model.run.completion_config) for task_id in problems]
                print(f"==== Completions generated in {time.time() - start} seconds ====")
        elif is_openai:
            args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
            llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
            model = GPT4Agent(llm=llm, **args.model.run.completion_config)
            completions = model.batch_prompt(
                system_message=data[0]['content'], 
                messages=[data[1]['content']] * args.model.run.batch_size,
            )
        else:
            print(f"Model type {args.model.model_type} not yet supported.")


    write_jsonl("samples.jsonl", samples)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass