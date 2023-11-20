import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any

from printllama.models.codellama import CodeLlama
from printllama.helpers import extract_code
from printllama.models.codellama_meta import Llama


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def generate_print_statements(question: str, faulty_solution: str, codellama: CodeLlama, n_prints: int) -> List[str]:
    """Generate print statements for debugging a given solution."""
    
    
    system = """Provide answers in Python."""

    user = f"""{question}
Here is a solution to the problem:
```python
{faulty_solution}
```
Insert print statements within in the initial solution to the problem that will help me debug and improve the program.
Do not add anything else, only insert print statements WITHIN THE INITIAL SOLUTION that will help me debug and improve the program"""

    prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user}{E_INST}"
    codellama_prints = []
    codellama_responses = []
    generator = Llama.build(
        ckpt_dir="/scr/jphilipp/printllama-hgx/codellama/CodeLlama-7b-Instruct",
        tokenizer_path="/scr/jphilipp/printllama-hgx/codellama/CodeLlama-7b-Instruct/tokenizer.model",
        max_seq_len=2000,
        max_batch_size=4,
    )
    breakpoint()
    for n_print in range(n_prints):
        # breakpoint()
        print(f"Generating print statement: {n_print + 1}/{n_prints}")
        codellama_print = codellama(prompt)

        codellama_responses.append(codellama_print)
        try:
            codellama_prints.append(extract_code([codellama_print]))
        except:
            print(f"Error in generating print statement, appending failed statement to list.")
            codellama_prints.append("Failed to generate print statement.")
    return codellama_prints, codellama_responses

def add_codellama_prints(args: argparse.Namespace, codellama: CodeLlama) -> None:
    """Add generated print statements to the dataset and overwrite it."""
    ds = load_dataset('json', data_files=args.dataset_path)
    updated_items: List[Dict[str, Any]] = []

    for item in tqdm(ds['train']):
        question = item['question']
        faulty_solution = item['faulty_solutions'][0]
        codellama_prints, codellama_responses = generate_print_statements(question, faulty_solution, codellama, args.n_prints)
        item['codellama_print_statements'] = codellama_prints
        item['codellama_responses'] = codellama_responses
        updated_items.append(item)

    with open(args.dataset_path, 'w') as f:
        json.dump(updated_items, f, indent=4)

def main() -> None:
    # Dataset args
    parser = argparse.ArgumentParser(description="Process dataset items by adding CodeLlama-generated print statements.")
    parser.add_argument("--dataset_path", type=str, default="../../data/filtered_apps_introductory.json", help="Path to the dataset file.")
    parser.add_argument("--n_prints", type=int, default=5, help="Number of prints to add.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    # Llama args
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--load_in_8bit", type=str, default=False)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--model_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf")
    parser.add_argument("--tokenizer_cache_dir", type=str, default="/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--use_flash_attention_2", type=bool, default=True)
    
    args = parser.parse_args()

    codellama = CodeLlama(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        use_flash_attention_2=args.use_flash_attention_2,
        max_new_tokens=args.max_new_tokens)

    add_codellama_prints(args, codellama)

if __name__ == "__main__":
    main()