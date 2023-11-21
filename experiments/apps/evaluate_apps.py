import multiprocessing
import argparse
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import testing_util_apps as test_util
import doctest
import json


# Define _temp_run at the module level
def _temp_run(test, input_output_pairs, debug, result):
    result.append(test_util.run_test(test, input_output_pairs, debug))

def check_correctness(solution_code, input_output_pairs, timeout, debug):
    """Check correctness of code generation with a global timeout."""
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(solution_code, input_output_pairs, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        p.join()
        return -1  # Indicates timeout

    return result[0] if result else -2  # Indicates error or no result

def eval_and_save_problems(questions, solution, debug=False):
    accuracies = []

    for i, question in enumerate(tqdm(questions)):
        solution_code = question[solution][0]
        if isinstance(solution_code, list):
            solution_code = solution_code[0]
        input_output_pairs = eval(question['input_output'])
        result = check_correctness(solution_code, input_output_pairs, timeout=10, debug=debug)
        try:
            accuracy = sum(1 for r in result if r == True) / len(result)
        except:
            accuracy = 0.0
        print(f"Question {i} Accuracy: {accuracy}")
        accuracies.append(accuracy)
        
    return accuracies


def main():
    parser = argparse.ArgumentParser(description='Evaluate solutions for coding problems')
    parser.add_argument('--data_path', type=str, default="../../data/apps_100_llama_prints_1_per_item_corrected.json", help='The path to the dataset')
    parser.add_argument('--n_questions', type=int, default=100, help='Number of questions to evaluate')
    parser.add_argument('--solution', type=str, default="codellama_print_statements", help='The solution to evaluate')
    parser.add_argument('--data_path_save', type=str, default="../../data/apps_100_llama_prints_1_per_item_corrected_w_acc.json", help='The path to store the results')

    args = parser.parse_args()

    # Load dataset
    ds = load_dataset('json', data_files=args.data_path)

    questions = [ds['train'][i] for i in range(len(ds['train']))][:args.n_questions]

    doctest.testmod()  
    accuracies = eval_and_save_problems(questions, solution=args.solution, debug=True)

    ds_results = []

    for i, item in enumerate(tqdm(ds['train'].select(range(args.n_questions)))):
        new_item = item.copy()  # Create a copy of the item
        new_item[f"{args.solution}_accuracies"] = accuracies[i]
        ds_results.append(new_item)

    updated_ds = Dataset.from_pandas(pd.DataFrame(ds_results))
    updated_ds.to_json(args.data_path_save)
        


if __name__ == "__main__":
    main()
