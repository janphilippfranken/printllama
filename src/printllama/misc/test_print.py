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
    max_seq_len: int = 1024,
    max_batch_size: int = 50,
    max_gen_len: Optional[int] = None,
    n_batch: int = 50,
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
    print(f"Dimensions of m, n, k: {m, n, k}")
    p = B.shape[1]
    A_sliced = A[slice_index, :, :]
    print(f"Shape of A_sliced: A_sliced.shape")
    result = torch.mm(A_sliced, B)
    print(f"Current result shape: {result.shape}")
    return result.view(m, 1, p)  
```

Input: 
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)
slice_index = -1

Print Output:
Dimensions of m, n, k: (3, 4, 5)
Shape of A_sliced: torch.Size([4, 5])
Current result shape: torch.Size([4, 6])

You are not allowed to use external libraries besides torch.

Format your improved solution as follows:
```python
def algorithm(A, B, slice_index):
    # Your code here
```

Important: Only return ONE solution and make sure that it strictly adheres to the format above. Do not return any other code. Only return the improved function called "algorithm".""",
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

    m, n, k = 3, 4, 5
    p = 6
    A = torch.randn(m, n, k)
    B = torch.randn(k, p)
    slice_index = -1


   
    for instruction, result in zip(instructions, results):

        try:

            code = extract_code(result["generation"]["content"])
            print(code)
            exec(code, globals())
            code_result =  algorithm(A, B, slice_index).shape
            evals.append(code_result == (m, 1, p))
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
    breakpoint()

if __name__ == "__main__":
    fire.Fire(main)