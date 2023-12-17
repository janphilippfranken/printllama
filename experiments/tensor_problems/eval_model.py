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

from printllama.helpers import extract_code


# import models
from printllama.models.codellama_meta.generation import Llama
from printllama.models.huggingface.hf_inference_model import HFInferenceModel

# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Running inference model...")


    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model.model_type.lower()
    is_hf = "hf" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()


    # GET DATA
    start = time.time()
    with open(args.data.data_path, "r") as f:
        data = json.load(f)
    print(f"==== Data loaded in {time.time() - start} seconds ====")


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
            batched_prompts = [f"{data[0]['content']}\n\n{data[1]['content']}"] * args.model.run.batch_size
            completions = model.batch_prompt(batched_prompts, **args.model.run.completion_config)
        else:
            print(f"Model type {args.model.model_type} not yet supported.")


    # COMPUTE METRICS FOR EXAMPLE UNIT TEST (currently hardcoded)
    responses = [] 
    results = []
    evals_count = 0
    

    # TODO: streamline the unit testing structure
    if 'attentionproblem' in args.data.data_path:
        # d_Q, d_K must be equal
        d_Q, d_K, d_V = 32, 32, 48
        batch_size = 64

        # input_len == output_len -> can be treated as self attention
        # input_len != output_len -> cross attention
        input_len, output_len = 50, 50  

        Q, K, V = torch.randn(batch_size, input_len, d_Q), torch.randn(batch_size, output_len, d_K), torch.randn(batch_size, output_len, d_V)
        # in cross attention, K and V are for sequence B, Q is for sequence A
    

    # LOAD SOLUTION
    start = time.time()
    with open(args.data.solution_path, 'r') as f:
        solution = f.read()
        solution = re.sub('\n', '\\n', solution)  
        exec(solution, globals())   ## set Solution.algorithm() to the solution function
    print(f"==== Solution loaded in {time.time() - start} seconds ====")


    # EVALUATE COMPLETIONS
    start = time.time()
    for completion in completions:
        try:
            if is_meta:
                responses.append(completion["generation"]["content"])
                code = extract_code(completion["generation"]["content"])
            
            
            exec(code, globals())  ## set output of completion to the function to be tested
            if 'attentionproblem' in args.data.data_path:
                correct = Solution.algorithm(Q, K, V).shape == algorithm(Q, K, V).shape
                results.append(correct)
            evals_count += 1
        except:
            results.append(False)
        
        # remove algorithm from memory to not contaminate results with incorrectly named outputs
        globals().pop('algorithm', None)  
    print(f"==== Evaluation completed in {time.time() - start} seconds ====")


    # PRINT METRICS
    print(f"Overall accuracy: {sum(results) / len(results)}")
    print(f"Evaluated {evals_count} out of {args.model.run.batch_size} unit tests")
    if evals_count: print(f"Accuracy among evaluatable outputs: {sum(results) / evals_count}")


    
    
    model_name = args.model.name
    data_name = args.data.name
    output_path = model_name + '_' + data_name
    
    
    if not os.path.exists(f'completions/{args.data.problem_name}/{args.model.name}/seed{seed}/'):
        os.makedirs(f'completions/{args.data.problem_name}/{args.model.name}/seed{seed}/')
    if not os.path.exists(f'metrics/{args.data.problem_name}/{args.model.name}/seed{seed}/'):
        os.makedirs(f'metrics/{args.data.problem_name}/{args.model.name}/seed{seed}/')





    # WRITE COMPLETIONS TO FILE
    start = time.time()
    with open(f'completions/{args.data.problem_name}/{args.model.name}/seed{seed}/{args.data.problem_name}_completions.json', "w") as f:
        json.dump(completions, f)
    print(f"==== Completions written to file in {time.time() - start} seconds ====")


    # WRITE METRICS TO FILE
    metrics = {'ID' : output_path,
            'Overall accuracy' : sum(results) / len(results),
            'Evaluated # out of 100' : evals_count,
            'Accuracy among evaluatable outputs' : sum(results) / evals_count if evals_count else None}
    start = time.time()
    with open(f'metrics/{args.data.problem_name}/{args.model.name}/seed{seed}/{args.data.problem_name}_metrics.json', "w") as f:
        json.dump(metrics, f)
    print(f"==== Metrics written to file in {time.time() - start} seconds ====")


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass