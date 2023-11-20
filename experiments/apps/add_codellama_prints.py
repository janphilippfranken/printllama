import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any

from printllama.models.codellama import CodeLlama
from printllama.helpers import extract_code, extract_assistant_completion

from datasets import Dataset
import pandas as pd

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
First, propose an idea, then implement it. Do not add anything else, only insert print statements."""
    
    prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user} {E_INST}"
    codellama_prints = []
    codellama_responses = []
    breakpoint()
    for n_print in range(n_prints):
        print(f"Generating print statement: {n_print + 1}/{n_prints}")
        codellama_print = codellama(prompt)
        codellama_responses.append(codellama_print)
        try:
            codellama_assistant_response = extract_assistant_completion(codellama_print)
            codellama_prints.append(extract_code([codellama_assistant_response]))
        except:
            print(f"Error in generating print statement, appending failed statement to list.")
            codellama_prints.append("Failed to generate print statement.")
    return codellama_prints, codellama_responses

def add_codellama_prints(args: argparse.Namespace, codellama: CodeLlama) -> None:
    ds = load_dataset('json', data_files=args.dataset_path_load)['train']

    updated_items = []

    for item in tqdm(ds):
        # Check if CodeLlama outputs already exist
        if 'codellama_print_statements' in item:
            updated_items.append(item)
            continue

        question = item['question']
        faulty_solution = item['faulty_solutions'][0]
        codellama_prints, codellama_responses = generate_print_statements(question, faulty_solution, codellama, args.n_prints)
        item['codellama_print_statements'] = codellama_prints
        item['codellama_responses'] = codellama_responses
        updated_items.append(item)
        breakpoint()
    updated_dataset = Dataset.from_pandas(pd.DataFrame(updated_items))
    updated_dataset.to_json(args.dataset_path_save)


def main() -> None:
    # Dataset args
    parser = argparse.ArgumentParser(description="Process dataset items by adding CodeLlama-generated print statements.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/repaired_apps_filtered_intro_test.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/codellama_apps_filtered_intro_test.json", help="Path to save the updated dataset file.")
    parser.add_argument("--n_prints", type=int, default=1, help="Number of prints to add.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

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