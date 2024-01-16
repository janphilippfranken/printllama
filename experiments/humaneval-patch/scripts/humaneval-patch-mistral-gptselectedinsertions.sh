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
python eval/eval_model.py data=humaneval-patch-011224-temp07-gpt4prints-exploded-selected-prints-gpt4 model=mistral-7b-instruct-v02-hf