from typing import List, Dict, Tuple
import os
import argparse
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset
import multiprocessing
import queue

from printllama.models.azure import AsyncAzureChatLLM
from printllama.models.gpt4 import GPT4Agent
from printllama.helpers import evaluate_solutions, merge_datasets, remove_comments
from gpt4_prompts import format_raw_solutions, generate_faulty_solutions, baseline_repair_solutions


def evaluate_solutions_wrapper(
    solution: str,
    input_output_pairs: List[Dict],
    return_queue: multiprocessing.Queue
) -> None:
    """Wraps the evaluate_solutions function to be used in a multiprocessing context.

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
    """Calls the evaluate_solutions function with a specified timeout.

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
        return return_queue.get_nowait()
    except queue.Empty:
        return (0, 0, [], ["No output from evaluate_solutions"])


def process_batch(
    questions: List[str],
    solutions: List[str],
    input_values: List[Dict],
    output_values: List[Dict],
    gpt4_format: GPT4Agent,
    gpt4_corrupt: GPT4Agent,
    gpt4_repair: GPT4Agent,
    start_index: int,
    ds: Dataset
) -> List[Dict]:
    """Processes a batch of questions and solutions.

    Args:
        questions: A list of questions to process.
        solutions: A list of corresponding solutions.
        input_values: Input values for the solutions.
        output_values: Expected output values for the solutions.
        gpt4_format: GPT4 formatting agent.
        gpt4_corrupt: GPT4 corruption agent.
        gpt4_repair: GPT4 repair agent.
        start_index: Start index for processing in the dataset.
        ds: The dataset to be updated.

    Returns:
        A list of updated dataset items.
    """
    formatted_solutions_batch = format_raw_solutions(solutions, input_values, gpt4_format)
    faulty_solutions_batch = generate_faulty_solutions(formatted_solutions_batch, input_values, output_values, gpt4_corrupt)
    faulty_solutions_batch = [remove_comments(solution) for solution in faulty_solutions_batch]
    baseline_repaired_batch = baseline_repair_solutions(questions, faulty_solutions_batch, gpt4_repair)

    updated_items = []

    for j, (formatted_solution, faulty_solution, repaired_solution) in enumerate(zip(formatted_solutions_batch, faulty_solutions_batch, baseline_repaired_batch)):
        current_item = ds[start_index + j]
        input_output_pairs = eval(current_item['input_output'])

        formatted_results = call_evaluate_solutions_with_timeout(formatted_solution, input_output_pairs, timeout=1)
        faulty_results = call_evaluate_solutions_with_timeout(faulty_solution, input_output_pairs, timeout=1)
        repaired_results = call_evaluate_solutions_with_timeout(repaired_solution, input_output_pairs, timeout=1)

        formatted_accuracy, formatted_deterministic_accuracy, formatted_prints, formatted_errors = formatted_results
        faulty_accuracy, faulty_deterministic_accuracy, faulty_prints, faulty_errors = faulty_results
        repaired_accuracy, repaired_deterministic_accuracy, repaired_prints, repaired_errors = repaired_results

        current_item['formatted_solution_accuracy'] = formatted_accuracy
        current_item['faulty_solution_accuracy'] = faulty_accuracy
        current_item['repaired_solution_accuracy'] = repaired_accuracy
        current_item['formatted_solution_deterministic_accuracy'] = formatted_deterministic_accuracy
        current_item['faulty_solution_deterministic_accuracy'] = faulty_deterministic_accuracy
        current_item['repaired_solution_deterministic_accuracy'] = repaired_deterministic_accuracy
        current_item['formatted_solutions'] = formatted_solution
        current_item['faulty_solutions'] = faulty_solution
        current_item['repaired_solutions'] = repaired_solution
        current_item['formatted_print_outputs'] = formatted_prints
        current_item['faulty_print_outputs'] = faulty_prints
        current_item['repaired_print_outputs'] = repaired_prints
        current_item['formatted_errors'] = formatted_errors
        current_item['faulty_errors'] = faulty_errors
        current_item['repaired_errors'] = repaired_errors

        updated_items.append(current_item)

    return updated_items


def format_dataset(
    args: argparse.Namespace,
    gpt4_format: GPT4Agent,
    gpt4_corrupt: GPT4Agent,
    gpt4_repair: GPT4Agent
) -> None:
    """Formats the dataset using GPT4 agents for formatting, corrupting, and repairing solutions.

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
    input_values = []
    output_values = []

    for i, item in enumerate(tqdm(ds)):
        if any(item['problem_id'] == result['problem_id'] for result in ds_results):
            print(f"Skipping item {i} because it is already in the results dataset.")
            continue
        
        questions.append(item['question'])
        solutions.append(item['correct_solution'])
        input_values.append(item['input_output'])
        output_values.append(item['input_output'])

        if len(solutions) == args.budget or i == len(ds) - 1:
            updated_items = process_batch(questions=questions,
                                          solutions=solutions,
                                          input_values=input_values,
                                          output_values=output_values,
                                          gpt4_format=gpt4_format,
                                          gpt4_corrupt=gpt4_corrupt,
                                          gpt4_repair=gpt4_repair,
                                          start_index=i - args.budget + 1, 
                                          ds=ds)
            ds_results = merge_datasets(ds_results, updated_items)
            
            questions = []
            solutions = []
            input_values = []
            output_values = []

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