import hydra
from omegaconf import DictConfig
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
import random
import re

from printllama.helpers import extract_code


# import models
from printllama.models.codellama_meta.generation import Llama
from printllama.models.huggingface.hf_inference_model import HFInferenceModel

# logging
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="config", config_name="codellama-7b-hf")
def main(args: DictConfig) -> None:
    logging.info("Running inference model...")

    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model_type.lower()
    is_hf = "hf" in args.model_type.lower()
    is_openai = "openai" in args.model_type.lower()

    # GET DATA
    with open(args.data.data_path, "r") as f:
        data = json.load(f)

    # BUILD MODEL AND RUN INFERENCE
    if not args.run.verbose:
        if is_meta:
            model = Llama.build(**args.model_config)
            batched_prompts = [data] * args.run.batch_size
            completions = model.chat_completion(
                batched_prompts,
                **args.run.completion_config,
            )
        elif is_hf: 
            model = HFInferenceModel(**args.model_config)
            batched_prompts = [f"{data[0]['content']}\n\n{data[1]['content']}"] * args.run.batch_size
            completions = model.batch_prompt(batched_prompts, **args.run.completion_config)
        else:
            print(f"Model type {args.model_type} not yet supported.")
    # COMPUTE METRICS FOR EXAMPLE UNIT TEST (currently hardcoded)
    responses = [] 
    results = []
    evals_count = 0


    if 'paddingproblem' in args.data.data_path:
        NUM_TENSORS = 16
        MIN_TENSOR_SIZE = 16
        MAX_TENSOR_SIZE = 256
        MAX_ELEMENT_VAL = 32000
        id = 32001
        tensors = [torch.randint(high=MAX_ELEMENT_VAL, size=(random.randint(MIN_TENSOR_SIZE, MAX_TENSOR_SIZE + 1),)) for _ in range(NUM_TENSORS)]
    else:
        m, n, k = 3, 4, 5
        p = 6
        A = torch.randn(m, n, k)
        B = torch.randn(k, p)
        slice_index = -1
    

    # load solution
    with open(args.data.solution, 'r') as f:
        solution = f.read()
        solution = re.sub('\n', '\\n', solution)
        exec(solution, globals())  ## set algorithm_correct() to the solution function


    for completion in completions:
        try:
            print(f"Globals within block: {globals()}")
            if is_meta:
                responses.append(completion["generation"]["content"])
                code = extract_code(completion["generation"]["content"])
            elif is_hf:
                responses.append(completion)
                code = extract_code(completion)
            exec(code, globals())
            if 'paddingproblem' in args.data.data_path:
                results.append(torch.equal(algorithm_correct(tensors, id), algorithm(tensors, id)))
            else:
                shape = algorithm(A, B, slice_index).shape
                results.append(shape == torch.Size([m * p]))
            evals_count += 1
        except:
            results.append(False)

    
    print(f"Accuracy: {sum(results) / len(results)}")
    print(f"Evaluated {evals_count} out of {args.run.batch_size} unit tests")

    # write completions to file
    breakpoint()
    with open(f'{args.model_type.lower()}_{args.data.data_path}_completions.json', "w") as f:
        json.dump(completions, f)

if __name__ == '__main__':
    fire.Fire(main())