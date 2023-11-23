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
        top_p=1.0,
        n=1, 
    )


batch_size = 50


system_message = """You are an expert researcher and programmer, especially skilled at debugging algorithms"""

human_message = """Repair the algorithm below. 

The algorithm takes three inputs:

1. A tensor A of shape (m, n, k).
2. A tensor B of shape (k, p).
3. An integer slice_index set to -1.


The algorithm should return a tensor of shape (m, 1, p) which is the result of a matrix multiplication between a correctly sliced tensor from A and tensor B.

Here is a current incorrect solution:
```python
import torch
def algorithm(A, B, slice_index):
    m, n, k = A.shape
    p = B.shape[1]
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(n, 1, p)

Input: 
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)
slice_index = -1
```

You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it. 
You algorithm has to run within max of 2 seconds and you are not allwed to use external libraries besides torch.

Format of the returned solution:
```python
def algorithm(A, B, slice_index):
    # Your code here
```

Just return code."""

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
    # breakpoint()
    try:
        exec(c, globals())
        result =  algorithm(A, B, slice_index).shape
        print(result)
        evals.append(result == (m, 1, p))
        counts_evaluated += 1
    except:
        evals.append(False)



system_message = """You are an expert researcher and programmer, especially skilled at debugging algorithms"""

human_message_prints = """Repair the algorithm below. 

The algorithm takes three inputs:

1. A tensor A of shape (m, n, k).
2. A tensor B of shape (k, p).
3. An integer slice_index set to -1.


The algorithm should return a tensor of shape (m, 1, p) which is the result of a matrix multiplication between a correctly sliced tensor from A and tensor B.

Here is a current incorrect solution:
```python
import torch
def algorithm(A, B, slice_index):
    m, n, k = A.shape
    p = B.shape[1]
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(n, 1, p)

Input: 
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)
slice_index = -1
```

You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it. 
You algorithm has to run within max of 2 seconds and you are not allwed to use external libraries besides torch.

Format of the returned solution:
```python
def algorithm(A, B, slice_index):
    # Your code here
```

Hint: I have previously tried to debug the solution and inserted the following print statments: 

``python
import torch
def algorithm_with_prints(A, B, slice_index):
    m, n, k = A.shape
    print(f"Dimensions of m, n, k: {m, n, k}")
    p = B.shape[1]
    A_sliced = A[slice_index, :, :]
    print(f"Shape of A_sliced: A_sliced.shape")
    result = torch.mm(A_sliced, B)
    print(f"Current result shape: {result.shape}")
    print(f"Current Output shape: {result.view(n, 1, p).shape}")
    return result.view(n, 1, p)
```

Which resulted in the following output:

```python
Dimensions of m, n, k: (3, 4, 5)
Shape of A_sliced: torch.Size([4, 5])
Current result shape: torch.Size([4, 6])
Current Output shape: (3, 1, 6)
```

Just return code."""

responses_print = gpt4.batch_prompt(system_message, [human_message_prints] * batch_size)
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
        evals_print.append(result == (m, 1, p))
        counts_evaluated_print += 1
    except:
        evals_print.append(False)



print(f"Mean evals without prints: {sum(evals) / len(evals)}", f"Number of evaluated solutions: {counts_evaluated}")
print(f"Mean evals with prints: {sum(evals_print) / len(evals_print)}", f"Number of evaluated solutions: {counts_evaluated_print}")

breakpoint()