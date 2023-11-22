import argparse
import os
from typing import List, Tuple
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

from printllama.models.codellama import CodeLlama
from printllama.helpers import extract_code, extract_assistant_completion, merge_datasets

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def generate_print_statements(
    question: str,
    faulty_solution: str,
    codellama: CodeLlama,
    n_prints: int
) -> Tuple[List[str], List[str]]:
    """Generate print statements for debugging a given solution using CodeLlama.

    Args:
        question: The problem statement.
        faulty_solution: The solution that needs debugging.
        codellama: An instance of the CodeLlama model.
        n_prints: Number of print statements to generate.

    Returns:
        A tuple containing two lists: one for the generated print statements and one for the responses.
    """
    system = """You are an expert computer science reasearcher and programmer, especially skilled at debugging algorithms."""

    user = f"""I have to solve the following problem:

{question}

Here is my initial solution to the problem:

```python
{faulty_solution}
```
Insert print statements within in the initial solution that will help me debug and improve the program. 
Be as creative as you can under the constraints. The return from your print statements must be helpful and non-trivial. 
First, propose an idea, then implement it. 

Important: Return the full initial solution as it is including your added print statemetns (within the solution). Do not return any other code.
Do not change any of the code in the intitial solution, only add helpful print statements."""
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
    print(codellama_prints)
    return codellama_prints, codellama_responses


def add_codellama_prints(
    args: argparse.Namespace, 
    codellama: CodeLlama,
) -> None:
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    # Load or initialize the results dataset
    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    updated_items = []

    for i, item in enumerate(tqdm(ds)):
        print(i, args.n_items, args.n_prints)
        if any(item['problem_id'] == result['problem_id'] and 'codellama_prints' in result for result in ds_results):
            continue

        question = item['question']
        faulty_solution = item['faulty_solutions']
        codellama_prints, codellama_responses = generate_print_statements(question, faulty_solution, codellama, args.n_prints)
        item['codellama_prints_7b'] = codellama_prints
        item['codellama_responses_7b'] = codellama_responses
        updated_items.append(item)

        ds_results = merge_datasets(ds_results, [item])

        updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
        updated_dataset.to_json(args.dataset_path_save)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process dataset items by adding CodeLlama-generated print statements.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/apps_intro_test_baseline.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/apps_intro_test_codellama.json", help="Path to save the updated dataset file.")
    parser.add_argument("--n_prints", type=int, default=10, help="Number of prints to add.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--n_items", type=int, default=100, help="Number of items to corrupt.")

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