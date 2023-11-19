import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

import os

from printllama.models.azure import AsyncAzureChatLLM
from printllama.models.gpt4 import GPT4Agent

from evaluate import check_correctness
from prompts import improve_solution, generate_faulty_solution

def process_dataset(args, gpt4_corrupt, gpt4_repair):
    # Load the dataset
    ds = load_dataset('json', data_files=args.dataset_path)
    updated_items = []

    for i, item in enumerate(tqdm(ds['train'])):

        question = item['question']
        correct_solution = item['correct_solution']

        # Generate faulty solution
        faulty_solutions = generate_faulty_solution(question, correct_solution, gpt4_corrupt)
        item['faulty_solutions'] = faulty_solutions

        # Generate improved solution
        improved_solutions = improve_solution(question, faulty_solutions[0], gpt4_repair)
        item['improved_faulty_solution_baseline'] = improved_solutions

        # Evaluate accuracies
        input_output_pairs = eval(item['input_output'])
        
        faulty_accuracy = check_correctness(faulty_solutions[0], input_output_pairs, args.timeout, args.debug)
        faulty_accuracy = sum(1 for r in faulty_accuracy if r == True) / len(faulty_accuracy)
        improved_accuracy = check_correctness(improved_solutions[0], input_output_pairs, args.timeout, args.debug)
        improved_accuracy = sum(1 for r in improved_accuracy if r == True) / len(improved_accuracy)

        item['faulty_solution_accuracy'] = faulty_accuracy
        item['improved_faulty_solution_accuracy'] = improved_accuracy

        updated_items.append(item)

    # Write the updated dataset to a new JSON file
    with open(args.dataset_path, 'w') as f:
        json.dump(updated_items, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty and repaired solutions.")
    parser.add_argument("--dataset_path", type=str, default="../../data/filtered_apps_introductory.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    # LLM Arguments 
    parser.add_argument("--azure_endpoint", type=str, default="https://philipp.openai.azure.com/")
    parser.add_argument("--api_version", type=str, default="2023-05-15")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--temperature_corrupt", type=float, default=0.7)
    parser.add_argument("--temperature_repair", type=float, default=0)
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

    gpt4_repair = GPT4Agent(
        llm=llm,
        budget=args.budget,
        model_id=args.model_id,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature_repair,
        top_p=args.top_p,
        n=args.n
    )


    process_dataset(args, gpt4_corrupt, gpt4_repair)

if __name__ == "__main__":
    main()
