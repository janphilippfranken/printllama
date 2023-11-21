from typing import List
import re

def extract_code(algorithm_strs: List[str]) -> List[str]:
    """Extract code from algorithm string."""
    extracted_code = []
    for algorithm_str in algorithm_strs:
        try:
            # Split the string by the code block delimiters
            code_block = algorithm_str.split("```")[1]
            
            code_block = re.sub(r"^\s*python\s*\n?", "", code_block, flags=re.IGNORECASE)

            extracted_code.append(code_block)
        except Exception:
            # Fallback code if extraction fails
            extracted_code.append("def algorithm(*args): return 0")

    return extracted_code


def extract_assistant_completion(completion: str) -> str:
    """
    Extracts the portion of the completion that comes after the [/INST] tag.
    """
    end_tag = "[/INST]"
    end_idx = completion.find(end_tag)

    if end_idx != -1:
        # Extracting the portion after the end tag
        post_inst_content = completion[end_idx + len(end_tag):]
        return post_inst_content.strip()  # Strips leading/trailing whitespace
    else:
        return ""  # Returns an empty string if the end tag is not found


def insert_prints_with_printllama(
    prompts, 
    printllama
    ) -> List[str]:
    """
    Inserts print statements using printllama peft adapter.
    """
    solutions = []
    for prompt in prompts:
        solutions.append(printllama(prompt))
    modified_solutions = extract_code(solutions)
    return modified_solutions

def insert_prints_with_codellama(
    prompts, 
    codellama
    ) -> List[str]:
    """
    Inserts print statements using codellama base model.
    """
    solutions = []
    for prompt in prompts:
        solutions.append(codellama(prompt))
    modified_solutions = extract_code(solutions)
    return modified_solutions

def insert_prints_with_gpt4(
    initial_solution, 
    gpt4) -> List[str]:
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

def repair_code_with_gpt4_and_prints(
    problem_description: str,
    initial_solution: str, 
    print_solution: str,
    prints: List[str],
    gpt4) -> List[str]:
    """
    Repair statements using azure gpt4.
    """
    system_message = "You are an expert computer science researcher and programmer, especially skilled at repairing algorithms."

    human_message = f"""I have to solve the following problem:

{problem_description}

Here is my initial solution to the problem:
```python
{initial_solution}
```
I have inserted the following print statements to debug the program:
```python
{print_solution}
```
Which have resulted in the following output:
```python
{prints}
```
Using the information from the print statements, please improve my solution. Do not include print statements in the improved solution."""
    solutions = gpt4.batch_prompt(system_message, [human_message] * gpt4.budget)        
    modified_solutions = extract_code(solutions)

    return modified_solutions

def format_llama_message(
        system_message: str, 
        human_message: str, 
        assistant_message: str
    ) -> str:
    """Format a message to fit Llama format. Based on:
        https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
        https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html).
    """
    return """[INST] <<SYS>>
{system_message}
<</SYS>>

{human_message} [/INST] {assistant_message}""".format(system_message=system_message, 
                                                      human_message=human_message, 
                                                      assistant_message=assistant_message)

def get_llama_messages(
    problem_description: str,
    initial_solution: str,
    assistant_response: str,
    ) -> str:
    """Get messages."""

    system_message = "You are an expert computer science researcher and programmer, especially skilled at debugging algorithms."

    human_message = f"""I have to solve the following problem:

{problem_description}

Here is my initial solution to the problem:
```python
{initial_solution}
```
Insert print statements in the program that will help me debug and improve the program."""
    assistant_message = f"""{assistant_response}"""
    return system_message, human_message, assistant_message

def tokenize_fn(
    text,
    tokenizer,
    max_length: int = 512,
    padding: str = "longest",
    return_tensors=None,
    truncation: bool = True,
    ignore_index: bool = False,
    ):
    result = tokenizer(
        text,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )
    if ignore_index:
        raise f"ignore_index not implemented; need to add -100 to labels"
    else:
        result["labels"] = result["input_ids"].copy()
    return result