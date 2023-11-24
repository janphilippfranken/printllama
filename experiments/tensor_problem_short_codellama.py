# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, List

import fire

from llama import Llama

import torch

import re


def extract_code(algorithm_str: List[str]) -> List[str]:
    """Extract code from algorithm string."""
    extracted_code = ""
    
    try:
        # Split the string by the code block delimiters
        code_block = algorithm_str.split("```")[1]
        code_block = re.sub(r"^\s*python\s*\n?", "", code_block, flags=re.IGNORECASE)
        extracted_code = code_block
    except Exception:
        # Fallback code if extraction fails
        extracted_code = "def algorithm(*args): return 0"

    return extracted_code

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 500,
    max_batch_size: int = 100,
    max_gen_len: Optional[int] = None,
    n_batch: int = 100,
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
                "content": """You are given a task and an incorrect solution to the task. Return an improved solution using the same function name and arguments.""",
            },
            {
                "role": "user",
                "content": """You are given a 3D tensor A of shape (3, 4, 5), a 2D tensor B of shape (5, 6), and an integer slice_index = -1. Slice A using slice_index at the correct dimension and multiply A_sliced with B to return a new 1D tensor of shape (3 * 6).

Incorrect Solution:
```python
import torch 

def algorithm(A, B, slice_index):
    A_sliced = A[slice_index, :, :]
    result = torch.mm(A_sliced, B)
    return result.view(-1)
```

You must return an improved solution. First, propose an idea, then implement it.""",
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
    evals_count = 0
    responses = []

    m, n, k = 3, 4, 5
    p = 6
    A = torch.randn(m, n, k)
    B = torch.randn(k, p)
    slice_index = -1

   
    for instruction, result in zip(instructions, results):
        try:
            responses.append(result["generation"]["content"])
            code = extract_code(result["generation"]["content"])
            exec(code, globals())
            code_result = algorithm(A, B, slice_index).shape
            evals.append(code_result == torch.Size([m * p]))
            evals_count += 1
        except:
            evals.append(False)
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    print(f"Accuracy: {sum(evals) / len(evals)}")
    print(f"Evaluated {evals_count} out of {len(instructions)} instructions")
    # write responses to text file 
    with open("responses.txt", "w") as f:
        for response in responses:
            f.write(response + "\n")

if __name__ == "__main__":
    fire.Fire(main)