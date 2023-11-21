import argparse
import doctest
from tqdm import tqdm
from datasets import load_dataset 
import pandas as pd
from datasets import Dataset
import io
import sys
import contextlib

def get_print_returns(solution_code, inputs):
    # Set up a string buffer to capture print outputs
    captured_output = io.StringIO()

    breakpoint()
    # A modified input function to provide predefined inputs
    def modified_input(_input_list):
        def input_stub():
            return _input_list.pop(0)
        return input_stub

    # Replace input calls in the solution code with predefined inputs
    input_replacement = modified_input(inputs)
    exec_globals = {
        'input': input_replacement,
        '__name__': '__main__'
    }

    with contextlib.redirect_stdout(captured_output):
        # Execute the solution code
        try:
            breakpoint()
            exec(solution_code, exec_globals)
        except Exception as e:
            # Handle exceptions gracefully
            return f"Error during execution: {e}"

    # Return the captured print outputs
    return captured_output.getvalue()



def eval_and_save_problems(questions, solution):
    print_returns = []

    for i, question in enumerate(tqdm(questions)):
        solution_code = question[solution][0]
        if isinstance(solution_code, list):
            solution_code = solution_code[0]
        # Assuming each question has at least one input-output pair
        input = eval(question['input_output'])['inputs'][0]
        print_output = get_print_returns(solution_code, input)
        print(f"Question {i} Print Returns: {print_output}")
        print_returns.append(print_output)
        breakpoint()
    return print_returns

def main():
    parser = argparse.ArgumentParser(description='Evaluate solutions for coding problems')
    parser.add_argument('--data_path', type=str, default="../../data/apps_100_llama_prints_1_per_item_corrected_w_acc.json", help='The path to the dataset')
    parser.add_argument('--n_questions', type=int, default=100, help='Number of questions to evaluate')
    parser.add_argument('--solution', type=str, default="codellama_print_statements", help='The solution to evaluate')
    parser.add_argument('--data_path_save', type=str, default="../../data/apps_100_llama_prints_1_per_item_corrected_w_prints.json", help='The path to store the results')

    args = parser.parse_args()

    # Load dataset
    ds = load_dataset('json', data_files=args.data_path)

    questions = [ds['train'][i] for i in range(len(ds['train']))][:args.n_questions]

    doctest.testmod()
    print_returns = eval_and_save_problems(questions, solution=args.solution)

    ds_results = []

    for i, item in enumerate(tqdm(ds['train'].select(range(args.n_questions)))):
        new_item = item.copy()
        new_item[f"{args.solution}_print_returns"] = print_returns[i]
        ds_results.append(new_item)

    updated_ds = Dataset.from_pandas(pd.DataFrame(ds_results))
    updated_ds.to_json(args.data_path_save)

if __name__ == "__main__":
    main()
