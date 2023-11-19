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

def eval_and_save_problems(questions, debug=False, data_path=None):
    results = {}
    questions_to_save = []

    for i, question in enumerate(tqdm(questions)):
        solution_code = eval(question['solutions'])[0]
        input_output_pairs = eval(question['input_output'])
        result = check_correctness(solution_code, input_output_pairs, timeout=10, debug=debug)
        results[i] = result
        accuracy = sum(1 for r in result if r == True) / len(result)
        print(f"Question {i} Accuracy: {accuracy}"
        if accuracy == 1:
            question["correct_solution_accuracy"] = accuracy
            question["correct_solution"] = solution_code
            question["correct_solution_idx"] = 0
            questions_to_save.append(question)
            print(accuracy)

    if data_path:
        with open(data_path, 'w') as f:
            json.dump(questions_to_save, f)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate solutions for coding problems')
    parser.add_argument('--difficulty', type=str, default='introductory', help='Difficulty level of problems')
    parser.add_argument('--n_questions', type=int, default=5, help='Number of questions to evaluate')

    args = parser.parse_args()

    # Modify data_path based on difficulty
    default_data_path = f"../../data/filtered_apps_{args.difficulty}.json"
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to save questions with 100% accuracy')

    args = parser.parse_args()  # Parse the arguments again to include data_path

    # Load dataset
    ds = load_dataset("codeparrot/apps", split="test")
    questions = [ds[i] for i in range(len(ds)) if ds[i]['difficulty'] == args.difficulty][:args.n_questions]

    doctest.testmod()  # This will run any doctests you have in your script
    eval_and_save_problems(questions, debug=True, data_path=args.data_path)

if __name__ == "__main__":
    main()
