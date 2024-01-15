import signal
import sys, os, io

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

MUTANTS_PROMPT_FORMAT = """Correct the following solution:
```python
{}
```

You will be evaluated based on the following evaluation function, which should run without error given your corrected solution as the 'candidate' function:
```python
{}
```
Your output should contain only the corrected code, without explanation or comments, keeping the original function name. Be as creative as you can under the constraints. Ensure the corrected Python code in your response is enclosed in triple backticks ``` ```.
"""

EXPERTISE = "You are an expert computer science researcher and programmer, especially skilled at fixing bugs in incorrect algorithms."


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class SuppressPrint:
    def __enter__(self):
        # Redirect standard output
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore standard output
        sys.stdout = self._original_stdout
    

import signal, time

class Timeout():
    """Timeout class using ALARM signal"""
    class Timeout(Exception): pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0) # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()
