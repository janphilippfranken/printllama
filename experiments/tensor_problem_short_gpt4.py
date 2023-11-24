from printllama.models.gpt4 import GPT4Agent
from printllama.models.azure import AsyncAzureChatLLM

import os

import argparse

from typing import List

from printllama.helpers import extract_code

import torch



llm = AsyncAzureChatLLM(
        api_key=os.getenv("OPENAI_API_KEY"), 
        azure_endpoint="https://philipp.openai.azure.com/",
        api_version="2023-05-15",
    )

gpt4= GPT4Agent(
        llm=llm,
        budget=1,
        model_id=1,
        model="gpt-4",
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
        n=1, 
    )


batch_size = 200


system_message = """You are given a task and an incorrect solution to the task. Return an improved solution using the same function name and arguments."""

human_message = """You are given a 3D tensor A of shape (3, 4, 5), a 2D tensor B of shape (5, 6), and an integer slice_index = -1. Slice A using slice_index at the correct dimension and multiply A_sliced with B to return a new 1D tensor of shape (3 * 6).

Incorrect Solution:
```python
import torch 

def algorithm(A, B, slice_index):
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(-1)
```

You must return an improved solution. First, propose an idea, then implement it."""

responses = gpt4.batch_prompt(system_message, [human_message] * batch_size)
code = extract_code(responses)

evals = []

m, n, k = 3, 4, 5
p = 6
A = torch.randn(m, n, k)
B = torch.randn(k, p)
slice_index = -1

counts_evaluated = 0

for c in code:
    try:
        exec(c, globals())
        result = algorithm(A, B, slice_index).shape
        evals.append(result == torch.Size([m * p]))
        counts_evaluated += 1
    except:
        evals.append(False)


system_message = """You are given a task, an incorrect solution to the task, and a debugging attempt including print statement returns. Return an improved solution using the same function name and arguments."""

human_message = """You are given a 3D tensor A of shape (3, 4, 5), a 2D tensor B of shape (5, 6), and an integer slice_index = -1. Slice A using slice_index at the correct dimension and multiply A_sliced with B to return a new 1D tensor of shape (3 * 6).

Incorrect Solution:
```python
import torch

def algorithm(A, B, slice_index):
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(-1)
```

Debugging Attempt:
```python
import torch

def algorithm(A, B, slice_index):
    print(f"Slice_index: {slice_index}")
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(-1)
```

Prints:
Slice_index: -1

You must return an improved solution. First, propose an idea, then implement it."""

responses_print = gpt4.batch_prompt(system_message, [human_message] * batch_size)
code_prints = extract_code(responses_print)

evals_print = []

m, n, k = 3, 4, 5
p = 6
A = torch.randn(m, n, k)
B = torch.randn(k, p)
slice_index = -1


counts_evaluated_print = 0

for c in code_prints:

    try:
        exec(c, globals())
        result = algorithm(A, B, slice_index).shape
        evals_print.append(result == torch.Size([m * p]))
        counts_evaluated_print += 1
    except:
        evals_print.append(False)


print(f"Mean evals without prints: {sum(evals) / len(evals)}", f"Number of evaluated solutions: {counts_evaluated}")
print(f"Mean evals with prints: {sum(evals_print) / len(evals_print)}", f"Number of evaluated solutions: {counts_evaluated_print}")

breakpoint()