# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama

import torch

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


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 10,
    max_gen_len: Optional[int] = None,
    n_batch: int = 10,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = n_batch * [[
            {
                "role": "system",
                "content": "You are an expert researcher and programmer, especially skilled at debugging algorithms",
            },
            {
                "role": "user",
                "content": """Repair the algorithm below. 

The algorithm takes three inputs:

1. A tensor A of shape (m, n, k).
2. A tensor B of shape (k, p).
3. An integer slice_index set to -1.


The algorithm should return a tensor of shape (m, 1, p) which is the result of a matrix multiplication between a correctly sliced tensor from A and tensor B.

Here is the current incorrect solution:
```python
import torch

def algorithm(A, B, slice_index):
    m, n, k = A.shape
    p = B.shape[1]
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(m, 1, p)  
```

Input: 
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)
slice_index = -1

You must return an improved solution. Be as creative as you can under the constraints.
Your primary improvement must be novel and non-trivial. First, propose an idea, then implement it. 
You algorithm has to run within max of 2 seconds and you are not allowed to use external libraries besides torch.

Format your improved solution as follows:
```python
def algorithm(A, B, slice_index):
    # Your code here
```""",
            }
        ]
]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    evals = []

    m, n, k = 3, 4, 5
    p = 6
    A = torch.randn(m, n, k)
    B = torch.randn(k, p)
    slice_index = -1


   
    for instruction, result in zip(instructions, results):
        try:
            code = extract_code(result["generation"]["content"])
            exec(code, globals())
            code_result =  algorithm(A, B, slice_index).shape
            evals.append(code_result == (m, 1, p))
        except:
            evals.append(False)
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    print(f"Accuracy: {sum(evals) / len(evals)}")


if __name__ == "__main__":
    fire.Fire(main)