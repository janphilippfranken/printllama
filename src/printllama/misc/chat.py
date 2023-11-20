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
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    breakpoint()

    instructions = [
        [
            {
                "role": "system",
                "content": "You are an expert computer science researcher and programmer, especially skilled at debugging algorithms.",
            },
            {
                "role": "user",
                "content": """I have to solve the following problem:
Recently you have received two positive integer numbers $x$ and $y$. You forgot them, but you remembered a shuffled list containing all divisors of $x$ (including $1$ and $x$) and all divisors of $y$ (including $1$ and $y$). If $d$ is a divisor of both numbers $x$ and $y$ at the same time, there are two occurrences of $d$ in the list.

For example, if $x=4$ and $y=6$ then the given list can be any permutation of the list $[1, 2, 4, 1, 2, 3, 6]$. Some of the possible lists are: $[1, 1, 2, 4, 6, 3, 2]$, $[4, 6, 1, 1, 2, 3, 2]$ or $[1, 6, 3, 2, 4, 1, 2]$.

Your problem is to restore suitable positive integer numbers $x$ and $y$ that would yield the same list of divisors (possibly in different order).

It is guaranteed that the answer exists, i.e. the given list of divisors corresponds to some positive integers $x$ and $y$.


-----Input-----

The first line contains one integer $n$ ($2 \le n \le 128$) — the number of divisors of $x$ and $y$.

The second line of the input contains $n$ integers $d_1, d_2, \dots, d_n$ ($1 \le d_i \le 10^4$), where $d_i$ is either divisor of $x$ or divisor of $y$. If a number is divisor of both numbers $x$ and $y$ then there are two copies of this number in the list.


-----Output-----

Print two positive integer numbers $x$ and $y$ — such numbers that merged list of their divisors is the permutation of the given list of integers. It is guaranteed that the answer exists.


-----Example-----
Input
10
10 2 8 1 2 4 1 20 4 5

Output
20 8
Here is my initial solution to the problem:
```python

n = int(input())

seq = sorted(list(map(int, input().split())))

a = seq[n-1]
last = -1
for i in range(len(seq)):
    if a % seq[i] != 0:
        if last != seq[i]:
            last = seq[i]
        else:
            b = seq[i]
            break
    else:
        b = seq[i]
        break
print(b, a)
```
Insert print statements within in the initial solution to the problem that will help me debug and improve the program.
Do not add anything else, only insert print statements WITHIN THE INITIAL SOLUTION that will help me debug and improve the program"""
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