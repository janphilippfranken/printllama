# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 2000,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {   "role": "system",
                "content": "provide answers in python",
            },
            {
                "role": "user",
                "content": """Insert print statements into the following python function called algorithm to debug the program and return the full algorithm including print statements as python code:
```python
import numpy as np

def algorithm(train_samples, train_parity, test_samples):
    n = train_samples.shape[1]
    aug_matrix = np.hstack((train_samples, train_parity.reshape(1, -1)))
    for i in range(n):
        pivot = np.where(aug_matrix[i:, i] == 1)[0]
        if len(pivot) > 0:
            pivot_row = pivot[0] + i
            if pivot_row != i:
                aug_matrix[[i, pivot_row]] = aug_matrix[[pivot_row, i]]
            for j in range(i + 1, len(aug_matrix)):
                if aug_matrix[j, i] == 1:
                    aug_matrix[j] = (aug_matrix[j] - aug_matrix[i]) % 2
    true_bits = np.zeros(n, dtype=int)
    for i in reversed(range(n)):
        true_bits[i] = aug_matrix[i, n]
        for j in range(i + 1, n):
            if aug_matrix[i, j] == 1:
                true_bits[i] ^= true_bits[j]
    test_parity = np.sum(true_bits + test_samples, axis=1) % 2
    return test_parity
```""",
            }
            
        ],

    ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)