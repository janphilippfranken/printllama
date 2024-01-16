#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting two GPU
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
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
python eval/eval_model.py data=humaneval-patch-011224-temp07-gpt4prints-exploded-selected-prints-gpt4 model=mixtral-8x7b-instruct-vllm