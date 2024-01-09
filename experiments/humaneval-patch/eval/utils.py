import signal

PROMPT_FORMAT = """Correct the following solution:
```python
{}
```

You will be evaluated based on the following evaluation function, which should run without error given your corrected solution as the 'candidate' function:
```python
{}
```
Your output should contain only the corrected code, without explanation or comments, keeping the original function name {}. Be as creative as you can under the constraints. Ensure the corrected Python code in your response is enclosed in triple backticks ``` ```.
"""

EXPERTISE = "You are an expert computer science researcher and programmer, especially skilled at fixing bugs in incorrect algorithms."


def signal_handler(signum, frame):
    raise Exception("Timed out!")