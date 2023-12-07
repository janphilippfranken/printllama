import hydra
from omegaconf import DictConfig
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch

from printllama.helpers import extract_code


# import models
from printllama.models.codellama_meta.generation import Llama
from printllama.models.huggingface.hf_inference_model import HFInferenceModel

# logging
logging.basicConfig(level=logging.INFO)

#### UNDO THIS WHEN DONE TESTING
@hydra.main(version_base=None, config_path="config", config_name="codellama-instruct-meta")
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
            batched_prompts = [data] * args.run.batch_size
            completions = model.batch_prompt(batched_prompts, **args.run.completion_config)
        else:
            print(f"Model type {args.model_type} not yet supported.")

    # COMPUTE METRICS FOR EXAMPLE UNIT TEST (currently hardcoded)
    responses = [] 
    results = []
    evals_count = 0

    # unit test params
    m, n, k = 3, 4, 5
    p = 6
    A = torch.randn(m, n, k)
    B = torch.randn(k, p)
    slice_index = -1

    for completion in completions:
        try:
            responses.append(completion["generation"]["content"])
            code = extract_code(completion["generation"]["content"])
            exec(code, globals())
            shape = algorithm(A, B, slice_index).shape
            results.append(shape == torch.Size([m * p]))
            evals_count += 1
        except:
            results.append(False)
    
    print(f"Accuracy: {sum(results) / len(results)}")
    print(f"Evaluated {evals_count} out of {args.run.batch_size} unit tests")

    # write completions to file
    breakpoint()
    with open('completions.json', "w") as f:
        json.dump(completions, f)

if __name__ == '__main__':
    fire.Fire(main())