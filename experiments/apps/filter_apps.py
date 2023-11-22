from typing import Dict, List

import multiprocessing
import argparse
from datasets import load_dataset
from tqdm import tqdm
import testing_util_apps as test_util
import doctest
import json



def _temp_run(
        test: str, 
        input_output_pairs: Dict[str, List],
        debug: bool,
        result: List[bool],
) -> None:
    """Run a single test with a global timeout."""
    result.append(test_util.run_test(test, input_output_pairs, debug))


def check_correctness(
    solution_code: str, 
    input_output_pairs: Dict[str, List], 
    timeout: int, 
    debug: bool
) -> int:
    """Check the correctness of code generation within a specified timeout."""
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(solution_code, input_output_pairs, debug, result))
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.terminate()
        p.join()
        return -1 # timeout

    return result[0] if result else -2  # error


def eval_and_save_problems(
    questions: List[Dict], 
    debug: bool = False, 
    data_path: str = None
) -> Dict:
    """Evaluate and save problem solutions based on their correctness."""
    results = {}
    questions_to_save = []

    for i, question in enumerate(tqdm.tqdm(questions)):
        try:
            solution_code = eval(question['solutions'])[0]
            input_output_pairs = eval(question['input_output'])
            result = check_correctness(solution_code, input_output_pairs, timeout=10, debug=debug)
            results[i] = result
            accuracy = sum(r == True for r in result) / len(result)
            print(f"Question {i} Accuracy: {accuracy:.2f}")

            if accuracy == 1:
                question.update({
                    "correct_solution_accuracy": accuracy,
                    "correct_solution": solution_code,
                    "correct_solution_idx": 0
                })
                questions_to_save.append(question)

        except Exception as e:
            print(f"Error on question {i}: {e}")

    if data_path:
        with open(data_path, 'w') as f:
            json.dump(questions_to_save, f)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate solutions for coding problems')
    parser.add_argument('--difficulty', type=str, default='introductory', help='Difficulty level of problems')
    parser.add_argument('--n_questions', type=int, default=2639, help='Number of questions to evaluate')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on')
    parser.add_argument('--data_path', type=str, help='Path to save questions with 100% accuracy')

    args = parser.parse_args()

    if not args.data_path:
        args.data_path = f"../../data/apps_{args.difficulty}_{args.split}.json"

    ds = load_dataset("codeparrot/apps", split=args.split)
    questions = [ds[i] for i in range(len(ds)) if ds[i]['difficulty'] == args.difficulty][:args.n_questions]

    doctest.testmod() 
    eval_and_save_problems(questions, debug=True, data_path=args.data_path)


if __name__ == "__main__":
    main()