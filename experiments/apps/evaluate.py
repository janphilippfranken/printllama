import multiprocessing
import argparse
from datasets import load_dataset
from tqdm import tqdm
import testing_util as test_util
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
    results = {}

    for i, question in enumerate(tqdm(questions)):
        solution_code = question[solution][0]
        input_output_pairs = eval(question['input_output'])
        result = check_correctness(solution_code, input_output_pairs, timeout=10, debug=debug)
        results[i] = result
        accuracy = sum(1 for r in result if r == True) / len(result)
        print(f"Question {i} Accuracy: {accuracy}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate solutions for coding problems')
    parser.add_argument('--data_path', type=str, default='filtered_apps_introductory', help='The path to the dataset')
    parser.add_argument('--n_questions', type=int, default=5, help='Number of questions to evaluate')
    parser.add_argument('--solution', type=str, default="improved_faulty_solution_baseline", help='The solution to evaluate')

    args = parser.parse_args()

    # Load dataset
    ds = load_dataset('json', data_files=args.data_path)

    questions = [ds['train'][i] for i in range(len(ds))][:args.n_questions]

    doctest.testmod()  # This will run any doctests you have in your script
    eval_and_save_problems(questions, solution=args.solution, debug=True)

if __name__ == "__main__":
    main()
