from typing import List, Tuple, Dict
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

def evaluate_solutions_wrapper(
    solution: str,
    input_output_pairs: List[Dict],
    return_queue: multiprocessing.Queue
) -> None:
    """Wraps the evaluate_solutions function for use in a multiprocessing context.

    Args:
        solution: The solution to be evaluated.
        input_output_pairs: A list of dictionaries containing input-output pairs for testing.
        return_queue: The queue to store the evaluation results.
    """
    try:    
        result = evaluate_solutions(solution, input_output_pairs)
        return_queue.put(result)
    except Exception as e:
        return_queue.put(e)

def call_evaluate_solutions_with_timeout(
    solution: str,
    input_output_pairs: List[Dict],
    timeout: int = 1
) -> Tuple:
    """Calls evaluate_solutions function with a specified timeout.

    Args:
        solution: The solution to be evaluated.
        input_output_pairs: A list of dictionaries containing input-output pairs for testing.
        timeout: Timeout in seconds for the evaluation process.

    Returns:
        A tuple containing the evaluation results or an error message.
    """
    return_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=evaluate_solutions_wrapper, args=(solution, input_output_pairs, return_queue))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return (0, 0, [], ["Timeout or resource limit exceeded"])

    try:
        return return_queue.get(timeout=2)
    except queue.Empty:
        return (0, 0, [], ["No output from evaluate_solutions"])

def process_batch(
    args: argparse.Namespace,
    questions: List[str],
    faulty_solutions: List[str],
    print_solutions: List[str],
    gpt4_repair: GPT4Agent,
    start_index: int,
    ds: Dataset
) -> List[Dict]:
    """Processes a batch of questions and solutions for repair.

    Args:
        args: Arguments received from the command line.
        questions: A list of questions to process.
        faulty_solutions: A list of faulty solutions.
        print_solutions: A list of solutions with prints.
        gpt4_repair: GPT4 repair agent.
        start_index: Start index for processing in the dataset.
        ds: The dataset to be updated.

    Returns:
        A list of updated dataset items.
    """
    print_returns_accuracy = []
    print_returns_deterministic_accuracy = []
    print_returns_prints = []
    print_returns_errors = []

    current_item = ds[start_index]
    input_output_pairs = eval(current_item['input_output'])
    
    for print_solution in print_solutions[0][:args.budget]:
        try:
            print_return = call_evaluate_solutions_with_timeout(print_solution[0], input_output_pairs, timeout=5)
            print_return_accuracy, print_return_deterministic_accuracy, print_return_prints, print_return_errors = print_return

            print_returns_accuracy.append(print_return_accuracy)
            print_returns_deterministic_accuracy.append(print_return_deterministic_accuracy)
            print_returns_prints.append(print_return_prints)
            print_returns_errors.append(print_return_errors)
        except Exception as e:
            print_returns_accuracy.append(0)
            print_returns_deterministic_accuracy.append(0)
            print_returns_prints.append([])
            print_returns_errors.append([str(e)])

    print_solutions_flat = [psol for lsol in print_solutions[0][:args.budget] for psol in lsol]

    print_repaired_batch = print_repair_solutions(questions[0], input_output_pairs, print_solutions_flat, print_returns_prints, faulty_solutions, gpt4_repair)
   

    updated_items = []
    print_repairs_accuracy = []
    print_repairs_deterministic_accuracy = []
    print_repairs_prints = []
    print_repairs_errors = []

    for print_repair_solution in print_repaired_batch:
        print_repaired_results = call_evaluate_solutions_with_timeout(print_repair_solution, input_output_pairs, timeout=5)
        print_repair_accuracy, print_repair_deterministic_accuracy, print_repair_prints, print_repair_errors = print_repaired_results
        print_repairs_accuracy.append(print_repair_accuracy)
        print_repairs_deterministic_accuracy.append(print_repair_deterministic_accuracy)
        print_repairs_prints.append(print_repair_prints)
        print_repairs_errors.append(print_repair_errors)

    current_item['print_repairs_accuracy'] = print_repairs_accuracy
    current_item['print_repairs_deterministic_accuracy'] = print_repairs_deterministic_accuracy
    current_item['print_repairs_prints'] = print_repairs_prints
    current_item['print_repairs_errors'] = print_repairs_errors
    current_item['print_returns_accuracy'] = print_returns_accuracy
    current_item['print_returns_deterministic_accuracy'] = print_returns_deterministic_accuracy
    current_item['print_returns_prints'] = print_returns_prints
    current_item['print_returns_errors'] = print_returns_errors
    
    updated_items.append(current_item)

    return updated_items


def format_dataset(
    args: argparse.Namespace,
    gpt4_format: GPT4Agent,
    gpt4_corrupt: GPT4Agent,
    gpt4_repair: GPT4Agent
) -> None:
    """Formats the dataset using GPT4 agents for corrupting and repairing solutions.

    Args:
        args: Arguments received from the command line.
        gpt4_format: GPT4 formatting agent.
        gpt4_corrupt: GPT4 corruption agent.
        gpt4_repair: GPT4 repair agent.
    """
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    questions = []
    solutions = []
    faulty_solutions = []
    print_solutions = []

    for i, item in enumerate(tqdm(ds)):
        if any(item['problem_id'] == result['problem_id'] for result in ds_results):
            print(f"Skipping item {i} because it is already in the results dataset.")
            continue
        
        questions.append(item['question'])
        solutions.append(item['correct_solution'])
        faulty_solutions.append(item['faulty_solutions'])
        print_solutions.append(item['codellama_prints_7b'])
 
        updated_items = process_batch(args=args,
                                        questions=questions,
                                        faulty_solutions=faulty_solutions,
                                        print_solutions=print_solutions,
                                        gpt4_repair=gpt4_repair,
                                        start_index=i,
                                        ds=ds)

        ds_results = merge_datasets(ds_results, updated_items)

        questions = []
        solutions = []
 

        print(gpt4_format.total_inference_cost)
        print(gpt4_corrupt.total_inference_cost)
        print(gpt4_repair.total_inference_cost)
        updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
        updated_dataset.to_json(args.dataset_path_save)

def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty solutions.")
    parser.add_argument("--dataset_path_load", type=str, default="../../data/apps_intro_test_codellama.json", help="Path to the dataset file.")
    parser.add_argument("--dataset_path_save", type=str, default="../../data/apps_intro_test_printllama_0.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=1, help="Timeout for solution evaluation.")
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