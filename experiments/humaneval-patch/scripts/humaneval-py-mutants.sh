#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPU
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=~/research_projects/printllama/experiments/humaneval-patch/job_output.%j.out
#SBATCH --error=~/research_projects/printllama/experiments/humaneval-patch/job_output.%j.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate printllama

# install necessary env packages
cd ~/research_projects/printllama
pip install -e .

# navigate to python script parent directory
cd ~/research_projects/printllama/experiments/humaneval-patch

# Array of model types
model_types=("huggingfaceh4-zephyr-7b-beta-hf" "mistral-7b-instruct-v02-hf")

# Loop through model types
for model_type in "${model_types[@]}"
do
    echo "Running model: $model_type"
    torchrun --nproc_per_node 1 --master_port 0 eval_model-mutants.py model=$model_type
done
