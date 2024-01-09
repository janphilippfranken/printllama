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
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset

from utils import PROMPT_FORMAT, EXPERTISE, signal_handler
from printllama.helpers import extract_code


# import models
from printllama.models.codellama_meta.generation import Llama
from printllama.models.huggingface.hf_inference_model import HFInferenceModel
from printllama.models.openai.azure import AsyncAzureChatLLM
from printllama.models.openai.gpt4 import GPT4Agent
from printllama.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Running inference model...")
    seed = args.model.model_config.seed
    
    
    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model.model_type.lower()
    is_hf = "hf" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()
    is_vllm = "vllm" in args.model.model_type.lower()
    
    
    # preprocess
    df = pd.read_csv(args.data.path).head(1)
    if args.condition.type == 'print': df = df[df['bugtype'].str.contains("print")]

   
    accs = list()
    samples = list()
    
    
    if not os.path.exists(f'completions/{args.data.path[5:-4]}/{args.model.name}/'):
        os.makedirs(f'completions/{args.data.path[5:-4]}/{args.model.name}/')
    if not os.path.exists(f'metrics/{args.data.path[5:-4]}/{args.model.name}/'):
        os.makedirs(f'metrics/{args.data.path[5:-4]}/{args.model.name}/')


    # BUILD MODEL AND RUN INFERENCE
    if not args.model.run.verbose:
        if not (is_openai or is_hf or is_vllm):
            print(f"Model type {args.model.model_type} not yet supported.")
            return -1
        
        if is_openai:
            args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
            llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
            model = GPT4Agent(llm=llm, **args.model.run.completion_config)
        elif is_hf:
            model = HFInferenceModel(**args.model.model_config)
        elif is_vllm:
            model = VLLMInferenceModel(**args.model.model_config)
        
        
        start = time.time()
        for i, row in df.iterrows():
            if pd.isnull(row['bug']): continue
            if i % 50 == 0:
                print("Saving existing results...")
                with open(f'metrics/{args.data.path[5:-4]}/{args.model.name}/seed{seed}.json', 'w') as f:
                    json.dump(accs, f)
                with open(f'completions/{args.data.path[5:-4]}/{args.model.name}/seed{seed}.json', 'w') as f:
                    json.dump(samples, f)
            print(f"Task {row['task_id']}")
            task_start = time.time()
            
            
            message = PROMPT_FORMAT.format(row['bug'], row['test'], row['entry_point'])
            if is_openai:
                print(f"Began batch prompting...")
                completions = model.batch_prompt(
                    system_message=EXPERTISE, 
                    messages=[message] * args.model.run.batch_size,
                )
                completions = [extract_code(completion) for completion in completions]
            elif is_hf:
                print(f"Began batch prompting...")
                if 'mistral' in args.model.name.lower():
                    # MISTRAL INST SPECIAL TOKENS
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    completions = [extract_code(completion) for completion in model.batch_prompt([f'<s>{B_INST}{B_SYS}{EXPERTISE}{E_SYS}{message}{E_INST}'], **args.model.run.completion_config)]
                elif 'zephyr' in args.model.name.lower():
                    # ZEPHYR INST SPECIAL TOKENS
                    B_SYS = '<|system|>'
                    B_USER = '<|user|>'
                    B_ASSISTANT = '<|assistant|>'
                    completions = [extract_code(completion) for completion in model.batch_prompt([f'<s>{B_SYS}{EXPERTISE}</s>\n{B_USER}{message}</s>\n{B_ASSISTANT}'], **args.model.run.completion_config)]
            elif is_vllm:
                if 'mixtral' in args.model.name.lower():
                    # MISTRAL INST SPECIAL TOKENS
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    completions = [extract_code(completion) for completion in model.batch_prompt([f'<s>{B_INST}{B_SYS}{EXPERTISE}{E_SYS}{message}{E_INST}'], **args.model.run.completion_config)]
            samples.append(completions)
            
            
            exec(row['test'], globals())
            passed = list()
            for j, completion in enumerate(completions):
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(1)   # One seconds
                
                try:
                    exec(completion, globals())
                    check(globals()[row['entry_point']])
                    passed.append(True)
                except:
                    passed.append(False)
            
                signal.alarm(0)
            
            accs.append(sum(passed) / len(passed) if len(passed) > 0 else 0.0)
            print("Accuracy: ", accs[-1])
            print(f"Finished in {time.time() - task_start} seconds")
        print(f"==== Completions generated in {time.time() - start} seconds ====")
    
    
    with open(f'metrics/{args.data.path[5:-4]}/{args.model.name}/seed{seed}.json', 'w') as f:
        json.dump(accs, f)
    with open(f'completions/{args.data.path[5:-4]}/{args.model.name}/seed{seed}.json', 'w') as f:
        json.dump(samples, f)
    
    print(f"==== Finished testing perturbations from {args.data.path[5:-4]} for {args.model.name} ====")


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass