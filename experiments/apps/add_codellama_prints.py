import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

import os

from printllama.models.codellama import CodeLlama

from printllama.helpers import extract_code

def add_codellama_prints(args, codellama):
    # Load the dataset
    ds = load_dataset('json', data_files=args.dataset_path)
    updated_items = []

    for i, item in enumerate(tqdm(ds['train'])):

        question = item['question']
        faulty_solution = item['faulty_solutions'][0]

        # Generate prints
        codellama_prints = []

        prompt = f"""[INST] <<SYS>>
You are an expert computer science researcher and programmer, especially skilled at debugging algorithms.
<</SYS>>
I have to solve the following problem:
{question}

Here is my initial solution to the problem:
```python
{faulty_solution}
```
Insert print statements in the program that will help me debug and improve the program.
Do not change the code, only insert print statements that will help me debug and improve the program.
[/INST]"""
        for _ in range(args.n_prints):
            codellama_prints.append(codellama(prompt.format(question=question, faulty_solution=faulty_solution)))
        
        item['codellama_print_statements'] = extract_code(codellama_prints)

        updated_items.append(item)

    # Write the updated dataset to a new JSON file
    with open(args.dataset_path, 'w') as f:
        json.dump(updated_items, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty and repaired solutions.")
    parser.add_argument("--dataset_path", type=str, default="../../data/filtered_apps_introductory.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--n_prints", type=int, default=5, help="Number of prints to add.")

    # LLM Arguments 
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--load_in_8bit", type=str, default=True)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--model_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf")
    parser.add_argument("--tokenizer_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf")
    parser.add_argument("--max_new_tokens", type=int, default=2000)

    args = parser.parse_args()

    codellama = CodeLlama(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        max_new_tokens=args.max_new_tokens)

    add_codellama_prints(args, codellama)

if __name__ == "__main__":
    main()
