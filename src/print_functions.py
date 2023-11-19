from helpers import extract_code


def insert_prints_with_printllama(prompts, printllama):
    """
    Inserts print statements using printllama peft adapter.
    """
    solutions = []
    for prompt in prompts:
        solutions.append(printllama(prompt))
    modified_solutions = extract_code(solutions)
    return modified_solutions


def insert_prints_with_codellama(prompts, codellama):
    """
    Inserts print statements using codellama base model.
    """
    solutions = []
    for prompt in prompts:
        solutions.append(codellama(prompt))
    modified_solutions = extract_code(solutions)
    return modified_solutions


def insert_prints_with_azure_gpt4(initial_solution, gpt4):
    """
    Inserts print statements using azure gpt4.
    """
    system_message = "You are an expert computer science researcher and programmer, especially skilled at debugging algorithms."

    human_message = f"""You are given the following Python program:
```python
{initial_solution}
```
Insert print statements in the program that will help me debug and improve the program."""
    solutions = gpt4.batch_prompt(system_message, [human_message] * gpt4.budget)        
    modified_solutions = extract_code(solutions)

    return modified_solutions

