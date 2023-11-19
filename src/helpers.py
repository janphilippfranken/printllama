from typing import List

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

def get_messages(
    initial_solution: str,
    assistant_response: str,
    ) -> str:
    """Get messages."""

    system_message = "You are an expert computer science researcher and programmer, especially skilled at debugging algorithms."

    human_message = """You are given the following Python program:
```python
{initial_solution}
```
Insert print statements in the program that will help me debug and improve the program."""

    assistant_message = """{assistant_response}"""
    human_message = human_message.format(initial_solution=initial_solution)
    assistant_message = assistant_message.format(assistant_response=assistant_response)
    return system_message, human_message.format(initial_solution), assistant_message.format(assistant_response)

def extract_code(
        algorithm_strs: List[str],
    ) -> str:
        """Extract code from algorithm string."""
        try:
            code = [algorithm_str.split("```")[1][6:] for algorithm_str in algorithm_strs]
        except:
             code = ["def algorithm(*args): return 0" for _ in algorithm_strs]
        return code

def tokenize(
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