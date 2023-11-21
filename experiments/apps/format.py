import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

import os
import pandas as pd

from printllama.models.azure import AsyncAzureChatLLM
from printllama.models.gpt4 import GPT4Agent

from evaluate_apps import check_correctness
from gpt4_prompts import format_raw_solutions

from datasets import Dataset

def compare_outputs(actual, expected):
    """
    Compares the actual output from the function with the expected output string.
    
    :param actual: The actual output from the function, can be of any type.
    :param expected: The expected output as a string, possibly including line breaks.
    :return: True if the outputs match, False otherwise.
    """
    # Convert the expected output to the same format as the actual output
    # Here, assuming expected output is separated by spaces and line breaks
    expected_normalized = expected.strip().split()
    
    # Flatten the actual output if it's a tuple or list
    if isinstance(actual, (tuple, list)):
        actual_normalized = [str(item) for item in actual]
    else:
        actual_normalized = [str(actual)]

    # Compare the flattened, normalized actual output with the normalized expected output
    return actual_normalized == expected_normalized



def process_batch(solutions, input_values, gpt4_corrupt, args, start_index, ds):

    formatted_solutions_batch = format_raw_solutions(solutions, input_values, gpt4_corrupt)
    # corrupt 
    # include prints
    # solve corrupted


    updated_items = []

    for j, formatted_solution in enumerate(formatted_solutions_batch):
        current_item = ds[start_index + j]
        input_output_pairs = eval(current_item['input_output'])
        
        try:
            exec(formatted_solution, globals())
            results = []
            for input, output in zip(input_output_pairs['inputs'], input_output_pairs['outputs']):
                result = solution_algorithm(input)
                results.append(compare_outputs(result, output))
            formatted_accuracy = sum(1 for r in results if r == True) / len(results)
        except:
            formatted_accuracy = 0.0

        print(gpt4_corrupt.total_inference_cost)
        print(formatted_accuracy)

        current_item['formatted_solution_accuracy'] = formatted_accuracy
        current_item['formatted_solutions'] = formatted_solution
        updated_items.append(current_item)

    return updated_items

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

def corrupt_dataset(args, gpt4_corrupt):
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    # Load or initialize the results dataset
    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    solutions = []
    input_values = []
    for i, item in enumerate(tqdm(ds)):
        # Skip if already in results dataset
        if any(item['problem_id'] == result['problem_id'] for result in ds_results):
            continue

        solutions.append(item['correct_solution'])
        input_values.append(eval(item['input_output'])['inputs'][0])

        if len(solutions) == args.budget or i == len(ds) - 1:
            solution_items = process_batch(solutions, input_values, gpt4_corrupt, args, i - args.budget + 1, ds)
            # Merge the updated items into the results dataset
            ds_results = merge_datasets(ds_results, solution_items)
            solutions = []
            input_values = []

    # Convert the list back to a dataset and save
    updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
    updated_dataset.to_json(args.dataset_path_save)



def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty solutions.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/apps_filtered_introductory_test.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/corrupted_apps_filtered_introductory_formatted.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--n_items", type=int, default=5, help="Number of items to corrupt.")

    # LLM Arguments 
    parser.add_argument("--azure_endpoint", type=str, default="https://philipp.openai.azure.com/")
    parser.add_argument("--api_version", type=str, default="2023-05-15")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--temperature_corrupt", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1)

    args = parser.parse_args()

    llm = AsyncAzureChatLLM(
        api_key=os.getenv("OPENAI_API_KEY"), 
        azure_endpoint=args.azure_endpoint, 
        api_version=args.api_version,
    )
    
    gpt4_corrupt = GPT4Agent(
        llm=llm,
        budget=args.budget,
        model_id=args.model_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature_corrupt,
        top_p=args.top_p,
        n=args.n
    )

    corrupt_dataset(args, gpt4_corrupt)

if __name__ == "__main__":
    main()