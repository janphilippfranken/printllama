from typing import List

from printllama.helpers import extract_code

def format_raw_solutions(
    solutions: List[str], 
    input_values: List[str],
    language_model,
) -> List[str]:
    """
    Formats an apps solution to make it easier to test it.

    Format:
        ```python
        def solution_algorithm(input_value):
            <part to modify>
        ```
    """
    system_message = "You are an expert computer science educator and programmer, especially skilled at designing algorithms."
    human_messages = []
    
    for solution, input_value in zip(solutions, input_values):
        input_value = eval(input_value)['inputs'][0].replace('\n', '\\n')
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
    input_values: List[str],
    output_values: List[str],
    language_model,
) -> List[str]:
    """ 
    Corrupts a solution to an APPS question using GPT-4.
    """
    system_message = "You are a skilled coder, renowned for your ability to manipulate correct code solutions and turn them into wrong solutions that are impossible to solve without inserting print statements for rigurous debugging."
    human_messages = []
    for solution, input_value, output_value in zip(solutions, input_values, output_values):
        input_value = eval(input_value)['inputs'][0].replace('\n', '\\n')
        output_value = eval(output_value)['outputs'][0].replace('\n', '\\n')
        human_message = f"""You are given the following correct solution to a coding problem:
```python
{solution}
```

Create a new wrong solution that is INCORRECT, has 0% ACCURACY, and breaks. The introduced errors should provide a rigorous test in debugging and problem-solving for expert computer scientists who are trying to fix the code.
If they manage to fix it, I will hire them to work for me, and my stadards are exceptionally high. SO it is your job to make sure that the code is as difficult to fix as possible.

An example input for testing the solution is:
{input_value}

An example output for testing the solution is:
{output_value}


Your wrong solution MUST FAIL the above test case, i.e. the new output given the input must be different from the output above. 
You can introduce additional errors that make the code break half-way or simliar, make it difficult to find errors without rigorous debugging, etc.

Return the faulty solution using this format:

Note: You can not change the name of the solution_algorithm() function or its input_values; for example doing something like input_values = input_values.split('\\n') -> input_values = input_values.split(' ') is not allowed."""
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
    Improves a solution to an AAPS question. 
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
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it.
Return the correct solution using this format:
```python
    def solution_algorithm(input_value):
        <correct solution>
        return <correct solution returned as an int or a tuple of ints>
```
Make sure that the code is running if solution_algorithm(input_value) is called with the correct input and returns it. Do not use external libraries such as multiprocessing and make sure the algorithm runs within milliseconds."""
        human_messages.append(human_message)
    improved_solutions = language_model.batch_prompt(system_message, human_messages)
    improved_solutions = extract_code(improved_solutions)

    return improved_solutions

def print_repair_solutions(
    questions,
    print_statements,
    print_returns,
    solutions, 
    language_model,
) -> List[str]:
    """
    Improves a solution to an AAPS question. 
    """
    system_message = "You are an expert computer science reasearcher and programmer, especially skilled at optimizing algorithms."
    human_messages = []
    for question, solution, print_statement, print_return in zip(questions, solutions, print_statements, print_returns):
        human_message = f"""Given this programming question:
{question}
and its faulty solution:
```python
{solution}
```
Return an improved (correct) solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it.
Return the correct solution using this format:
```python
    def solution_algorithm(input_value):
        <correct solution>
        return <correct solution returned as an int or a tuple of ints>
```
Make sure that the code is running if solution_algorithm(input_value) is called with the correct input and returns it. Do not use external libraries such as multiprocessing and make sure the algorithm runs within milliseconds.


HINT: I have already tried debugging the solution and inserted the following print statements:
```python
{print_statement}
```

The output of the print statements is:
```python
{print_return}
```

You might find these print statements to make the solution better."""
        human_messages.append(human_message)
    improved_solutions = language_model.batch_prompt(system_message, human_messages)
    improved_solutions = extract_code(improved_solutions)

    return improved_solutions