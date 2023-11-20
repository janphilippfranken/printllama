import argparse
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from printllama.models.azure import AsyncAzureChatLLM
from printllama.models.gpt4 import GPT4Agent

from evaluate import check_correctness
from gpt4_prompts import repair_solution  # Assuming this is the correct function name

from datasets import Dataset

def process_batch(batch_questions, batch_faulty_solutions, gpt4_repair, args):
    repaired_solutions_batch = repair_solution(batch_questions, batch_faulty_solutions, gpt4_repair)
    updated_items = []

    for j, repaired_solution in enumerate(repaired_solutions_batch):
        try:
            repaired_accuracy = check_correctness(repaired_solution, input_output_pairs, args.timeout, args.debug)
            repaired_accuracy = sum(1 for r in repaired_accuracy if r == True) / len(repaired_accuracy)
        except:
            repaired_accuracy = 0.0

        updated_items.append({'repaired_solution_accuracy': repaired_accuracy, 'repaired_solutions': repaired_solution})

    return updated_items

# Merge datasets function remains the same
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

def repair_dataset(args, gpt4_repair):
    ds = load_dataset('json', data_files=args.dataset_path_load)['train'].select(range(args.n_items))

    if os.path.exists(args.dataset_path_save):
        ds_results = load_dataset('json', data_files=args.dataset_path_save)['train']
    else:
        ds_results = []

    batch_questions = []
    batch_faulty_solutions = []

    for i, item in enumerate(tqdm(ds)):
        if 'repaired_solutions' in item:
            continue

        batch_questions.append(item['question'])
        batch_faulty_solutions.append(item['faulty_solutions'][0])

        if len(batch_faulty_solutions) == args.budget or i == len(ds) - 1:
            batch_items = process_batch(batch_questions, batch_faulty_solutions, gpt4_repair, args)
            ds_results = merge_datasets(ds_results, batch_items)
            batch_questions = []
            batch_faulty_solutions = []

    updated_dataset = Dataset.from_pandas(pd.DataFrame(ds_results))
    updated_dataset.to_json(args.dataset_path_save)

def main():
    parser = argparse.ArgumentParser(description="Process dataset items by adding faulty and repaired solutions.")
    parser.add_argument("--dataset_path", type=str, default="../../data/filtered_apps_introductory.json", help="Path to the dataset file.")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for solution evaluation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--n_repairs", type=int, default=5, help="Number of repairs to improve upon each faulty solution.")

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
        budget=args.budget * args.n_repairs,
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
