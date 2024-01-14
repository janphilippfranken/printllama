#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPU
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
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

# Print condition
torchrun --nproc_per_node 1 --master_port 0 eval/eval_model.py data=humaneval-patch-print model=huggingfaceh4-zephyr-7b-beta-hf condition=print

# Control condition
torchrun --nproc_per_node 1 --master_port 0 eval/eval_model.py data=humaneval-patch-control model=huggingfaceh4-zephyr-7b-beta-hf condition=control