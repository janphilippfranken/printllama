from typing import List

import os
import argparse
from tqdm import tqdm

import pandas as pd

from datasets import Dataset, load_dataset

import multiprocessing
import queue


from printllama.models.azure import AsyncAzureChatLLM
from printllama.models.gpt4 import GPT4Agent

from printllama.helpers import evaluate_solutions, merge_datasets

from gpt4_prompts import print_repair_solutions


def evaluate_solutions_wrapper(solution, input_output_pairs, return_queue):
    try:
        result = evaluate_solutions(solution, input_output_pairs)
        return_queue.put(result)
    except Exception as e:
        return_queue.put(e)

def call_evaluate_solutions_with_timeout(solution, input_output_pairs, timeout=1):
    return_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=evaluate_solutions_wrapper, args=(solution, input_output_pairs, return_queue))
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return (0, 0, [], ["Timeout or resource limit exceeded"])
    try:
        return return_queue.get_nowait()
    except queue.Empty:
        return (0, 0, [], ["No output from evaluate_solutions"])

def process_batch(questions, faulty_solutions, print_solutions, gpt4_repair, start_index, ds) -> List:

    breakpoint()
    print_returns = []
    for print_solution in print_solutions:
        try:
            exec(print_solution, globals())
            print_return = call_evaluate_solutions_with_timeout([print_solution], input_output_pairs, timeout=1)
            print_returns.append(print_return[2][0])
        except Exception as e:
            print_returns.append(["Error in executing solution: {e}"])

    print_repaired_batch = print_repair_solutions(questions, print_solutions, print_returns, faulty_solutions, gpt4_repair)

    updated_items = []

    for j, print_repair_solution in enumerate(print_repaired_batch):
        current_item = ds[start_index + j]
        input_output_pairs = eval(current_item['input_output'])

    
        print_repaired_results = call_evaluate_solutions_with_timeout(print_repair_solution, input_output_pairs, timeout=1)
        print_repair_accuracy, print_repair_deterministic_accuracy, print_repair_prints, print_repair_errors = print_repaired_results

        current_item['print_repair_accuracy'] = print_repair_accuracy
        current_item['print_repair_deterministic_accuracy'] = print_repair_deterministic_accuracy
        current_item['print_repair_prints'] = print_repair_prints
        current_item['print_repair_errors'] = print_repair_errors
        
        updated_items.append(current_item)

    return updated_items


def format_dataset(args, gpt4_format, gpt4_corrupt, gpt4_repair):

    # load dataset
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    # Load or initialize the results dataset
    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    questions = []
    solutions = []
    faulty_solutions = []
    print_solutions = []

    for i, item in enumerate(tqdm(ds)):
        # Skip if already in results dataset
        if any(item['problem_id'] == result['problem_id'] for result in ds_results):
            print(f"Skipping item {i} because it is already in the results dataset.")
            continue
        
        questions.append(item['question'])
        solutions.append(item['correct_solution'])
        faulty_solutions.append(item['faulty_solutions'])
        # print_solutions.append(item['codellama_prints_13b'])
 

        if len(solutions) == args.budget or i == len(ds) - 1:
            # Process the batch
            updated_items = process_batch(questions=questions,
                                          solutions=solutions,
                                          faulty_solutions=faulty_solutions,
                                          print_solutions=print_solutions,
                                          gpt4_repair=gpt4_repair,
                                          start_index=i - args.budget + 1, 
                                          ds=ds)
            # Merge the updated items into the results dataset
            ds_results = merge_datasets(ds_results, updated_items)
            # Reset the batch
            questions = []
            solutions = []
 

        # Convert the list back to a dataset and save
        print(gpt4_format.total_inference_cost)
        print(gpt4_corrupt.total_inference_cost)
        print(gpt4_repair.total_inference_cost)
        updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
        updated_dataset.to_json(args.dataset_path_save)

def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty solutions.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/apps_intro_test.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/apps_intro_test_baseline.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--n_items", type=int, default=100, help="Number of items to corrupt.")

    # LLM Arguments 
    parser.add_argument("--azure_endpoint", type=str, default="https://philipp.openai.azure.com/")
    parser.add_argument("--api_version", type=str, default="2023-05-15")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--temperatures", type=List, default=[0, 0.5, 0])
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1)

    args = parser.parse_args()

    llm = AsyncAzureChatLLM(
        api_key=os.getenv("OPENAI_API_KEY"), 
        azure_endpoint=args.azure_endpoint, 
        api_version=args.api_version,
    )

    gpt4_format = GPT4Agent(
        llm=llm,
        budget=args.budget,
        model_id=args.model_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperatures[0],
        top_p=args.top_p,
        n=args.n
    )
    
    gpt4_corrupt = GPT4Agent(
        llm=llm,
        budget=args.budget,
        model_id=args.model_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperatures[1],
        top_p=args.top_p,
        n=args.n
    )

    gpt4_repair = GPT4Agent(
        llm=llm,
        budget=args.budget,
        model_id=args.model_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperatures[2],
        top_p=args.top_p,
        n=args.n
    )

    format_dataset(args, gpt4_format, gpt4_corrupt, gpt4_repair)

if __name__ == "__main__":
    main()