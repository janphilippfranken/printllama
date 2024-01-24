#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting one GPU
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate printllama

# install necessary env packages
cd ~/research_projects/printllama
pip install -e .

# navigate to python script parent directory
cd ~/research_projects/printllama/experiments/humaneval-patch

torchrun --nproc_per_node 1 experiments/humaneval-patch/eval/eval_model.py model=mistral-7b-instruct-v02-hf data=humaneval-py-mutants

torchrun --nproc_per_node 2 experiments/humaneval-patch/eval/eval_model.py model=mixtral-8x7b-instruct-vllm data=humaneval-py-mutants

torchrun --nproc_per_node 1 experiments/humaneval-patch/eval/eval_model.py model=mistral-7b-instruct-v02-hf data=humaneval-patch-control condition=control

torchrun --nproc_per_node 2 experiments/humaneval-patch/eval/eval_model.py model=mixtral-8x7b-instruct-vllm data=humaneval-patch-control condition=control

torchrun --nproc_per_node 1 experiments/humaneval-patch/eval/eval_model.py model=mistral-7b-instruct-v02-hf data=humaneval-patch-gpt4-prints-exploded-selected-prints-gpt4 condition=print

torchrun --nproc_per_node 2 experiments/humaneval-patch/eval/eval_model.py model=mixtral-8x7b-instruct-vllm data=humaneval-patch-gpt4-prints-exploded-selected-prints-gpt4 condition=print

torchrun --nproc_per_node 1 experiments/humaneval-patch/eval/eval_model.py model=mistral-7b-instruct-v02-hf data=humaneval-patch-gpt4-prints-exploded-selected-prints-mixtral-8x7b-instruct-vllm condition=print

torchrun --nproc_per_node 2 experiments/humaneval-patch/eval/eval_model.py model=mixtral-8x7b-instruct-vllm data=humaneval-patch-gpt4-prints-exploded-selected-prints-mixtral-8x7b-instruct-vllm condition=print