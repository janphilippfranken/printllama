from typing import List

from printllama.helpers import extract_code

def generate_faulty_solution(
    question,
    solution, 
    language_model,
) -> List[str]:
    """
    Corrupts a solution to a apps question using GPT-4.
    """
    system_message = "You are an expert computer science educator and programmer, renowned for your ability to design intricate coding challenges."

    human_message = f"""Given this programming question:

{question}

and its correct solution:
```python
{solution}
```

Create a version of the solution that is intentionally incorrect. The errors introduced 
should be substantial and challenging to fix, providing a rigorous exercise in debugging 
and problem-solving for students."""
    faulty_solutions = language_model.batch_prompt(system_message, [human_message] * language_model.budget)
    faulty_solutions = extract_code(faulty_solutions)

    return faulty_solutions


def improve_solution(
    question,
    solution, 
    language_model,
) -> List[str]:
    """
    Improves a solution to a question.
    """
    system_message = "You are an expert computer science reasearcher and programmer, especially skilled at optimizing algorithms."

    human_message = f"""Given this programming question:

{question}

and its solution:
```python
{solution}
```

Return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it."""
    improved_solutions = language_model.batch_prompt(system_message, [human_message] * language_model.budget)
    improved_solutions = extract_code(improved_solutions)

    return improved_solutions