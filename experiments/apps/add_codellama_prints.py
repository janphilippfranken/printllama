import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any

from printllama.models.codellama import CodeLlama
from printllama.helpers import extract_code, extract_assistant_completion

from datasets import Dataset
import pandas as pd

import os

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def generate_print_statements(question: str, faulty_solution: str, codellama: CodeLlama, n_prints: int) -> List[str]:
    """Generate print statements for debugging a given solution."""
    
    
    system = """You are an expert computer science reasearcher and programmer, especially skilled at debugging algorithms."""

    user = f"""I have to solve the following problem:
{question}
Here is my initial solution to the problem:
```python
{faulty_solution}
```
Insert print statements within in the initial solution that will help me debug and improve the program. 
Be as creative as you can under the constraints. The return from your print statements must be helpful and non-trivial. 
First, propose an idea, then implement it. Return the full initial solution as it is including your added print statemetns (within the solution)."""
    
    prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user} {E_INST}"
    codellama_prints = []
    codellama_responses = []
    for n_print in range(n_prints):
        print(f"Generating print statement: {n_print + 1}/{n_prints}")
        try: 
            codellama_print = codellama(prompt)
            codellama_responses.append(codellama_print)
            codellama_assistant_response = extract_assistant_completion(codellama_print)
            codellama_prints.append(extract_code([codellama_assistant_response]))
        except:
            print(f"Error in generating print statement, appending failed statement to list.")
            codellama_prints.append("Failed to generate print statement.")
            codellama_responses.append("Failed to generate print statement.")
    return codellama_prints, codellama_responses


def merge_datasets(original, updates):
    # Convert to a list for easier manipulation
    original_list = [item for item in original]
    update_ids = {item['problem_id']: item for item in updates}

    # Merge the updated items into the original dataset
    for i, item in enumerate(original_list):
        if item['problem_id'] in update_ids:
            original_list[i].update(update_ids[item['problem_id']])

    # Add any new items that weren't in the original dataset
    for item in updates:
        if item['problem_id'] not in [original_item['problem_id'] for original_item in original_list]:
            original_list.append(item)

    return original_list

def add_codellama_prints(args: argparse.Namespace, codellama: CodeLlama) -> None:
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    # Load or initialize the results dataset
    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    updated_items = []

    for i, item in enumerate(tqdm(ds)):
        print(i, args.n_items, args.n_prints)
        # Skip if CodeLlama outputs already exist in results dataset
        if any(item['problem_id'] == result['problem_id'] and 'codellama_print_statements' in result for result in ds_results):
            continue

        question = item['question']
        faulty_solution = item['faulty_solutions'][0]
        codellama_prints, codellama_responses = generate_print_statements(question, faulty_solution, codellama, args.n_prints)
        item['codellama_print_statements'] = codellama_prints
        item['codellama_responses'] = codellama_responses
        updated_items.append(item)

        # Merge the updated item into the results dataset
        ds_results = merge_datasets(ds_results, [item])

        # Convert the list back to a dataset and save after each item
        updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
        updated_dataset.to_json(args.dataset_path_save)

def main() -> None:
    # Dataset args
    parser = argparse.ArgumentParser(description="Process dataset items by adding CodeLlama-generated print statements.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/apps_100_llama.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/apps_100_llama_prints_10_per_item.json", help="Path to save the updated dataset file.")
    parser.add_argument("--n_prints", type=int, default=10, help="Number of prints to add.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--n_items", type=int, default=100, help="Number of items to corrupt.")

    # Llama args
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--load_in_8bit", type=str, default=False)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--model_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_instruct_hf")
    parser.add_argument("--tokenizer_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_instruct_hf")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    
    args = parser.parse_args()

    codellama = CodeLlama(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        max_new_tokens=args.max_new_tokens)

    add_codellama_prints(args, codellama)

if __name__ == "__main__":
    main()