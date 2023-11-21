from typing import List

from printllama.helpers import extract_code

def generate_faulty_solution(
    solution, 
    language_model,
) -> List[str]:
    """
    Corrupts a solution to a apps question using GPT-4. If budget > 1, uses a batch for a single solution.
    """
    system_message = "You are an expert computer science educator and programmer, renowned for your ability to design intricate coding challenges."

    human_message = f"""You are given the following solution to a coding problem:
```python
{solution}
```
Create a version of the solution that is intentionally incorrect. The errors introduced 
should be substantial and challenging to fix, providing a rigorous exercise in debugging 
and problem-solving for students. The new incorrect solution must be wrong and have 0% or close to 0% accuracy."""
    faulty_solutions = language_model.batch_prompt(system_message, [human_message] * language_model.budget)
    faulty_solutions = extract_code(faulty_solutions)

    return faulty_solutions

def format_raw_solutions(
    solutions: List[str], 
    input_values: List[str],
    language_model,
) -> List[str]:
    """
    Formats a solution to make it easier to test it.
    """
    system_message = "You are an expert computer science educator and programmer, especially skilled at designing algorithms."
    human_messages = []
    for solution, input_value in zip(solutions, input_values):
        input_value = input_value.replace("\n", "\\n")
        human_message = f"""You are given the following solution to a coding problem.
{solution}
```
An example input for testing the solution is:
{input_value}

Reformat the provided coding solution into a Python function named solution_algorithm() which takes input_values as its only parameter. This parameter will be a string with line breaks, like '3 3 7 5\\n0 0 4 6\\n0 0 7 4\\n', indicating the required inputs.

1. Integrate the original solution into solution_algorithm() without any modifications.
2. Ensure solution_algorithm() processes input_values, splitting it by line breaks and formatting each line correctly (e.g., converting to integers or strings as needed).
3. Handle any user inputs or interactive components within solution_algorithm().
4. Do NOT include any comments in solution_algorithm() (e.g., # this is a comment) or change anything else in the original solution.

All that should remain to be done after your edits is calling solution_algorithm() with input_values as its only parameter and getting the return that we have gotten from the original solution."""
        
        human_messages.append(human_message)
    formatted_solutions = language_model.batch_prompt(system_message, human_messages)
    formatted_solutions = extract_code(formatted_solutions)

    return formatted_solutions

def generate_faulty_solutions(
    solutions: List[str], 
    language_model,
) -> List[str]:
    """
    Corrupts a solution to a apps question using GPT-4. Uses a batch of different solutions.
    """
    system_message = "You are an expert computer science educator and programmer, renowned for your ability to design intricate coding challenges."
    human_messages = []
    for solution in solutions:
        human_message = f"""You are given the following solution to a coding problem:
{solution}
```
Create a version of the solution that is intentionally incorrect. The errors introduced 
should be substantial and challenging to fix, providing a rigorous exercise in debugging 
and problem-solving for students. The new incorrect solution must be wrong and have 0% or close to 0% accuracy."""
        human_messages.append(human_message)
    faulty_solutions = language_model.batch_prompt(system_message, human_messages)
    faulty_solutions = extract_code(faulty_solutions)

    return faulty_solutions

def baseline_repair_solutions(
    questions,
    solutions, 
    language_model,
) -> List[str]:
    """
    Improves a solution to a question. Submits a batch request.
    """
    system_message = "You are an expert computer science reasearcher and programmer, especially skilled at optimizing algorithms."
    human_messages = []
    for question, solution in zip(questions, solutions):
        human_message = f"""Given this programming question:
{question}
and its faulty solution:
```python
{solution}
```
Return an improved (correct) solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""
        human_messages.append(human_message)
    improved_solutions = language_model.batch_prompt(system_message, human_messages)
    improved_solutions = extract_code(improved_solutions)

    return improved_solutions

def print_repair_solutions(
    questions,
    solutions, 
    print_solutions,
    print_returns,
    language_model,
) -> List[str]:
    """
    Improves a solution to a question given prints. Submits a batch request.
    """
    system_message = "You are an expert computer science reasearcher and programmer, especially skilled at optimizing algorithms."
    human_messages = []
    for question, solution, print_solution, print_return, in zip(questions, solutions, print_solutions, print_returns):
        human_message = f"""You are given this programming question:
{question}
and its faulty solution:
```python
{solution}
```

To improve this solution, consider the following print statements used for debugging:
```python
{print_solution}
```
The output from these print statements was:
```
{print_return}
```

Return an improved (correct) solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""
        human_messages.append(human_message)
    improved_solutions = language_model.batch_prompt(system_message, human_messages)
    improved_solutions = extract_code(improved_solutions)

    return improved_solutions