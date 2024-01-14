#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting one GPU
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

# Array of model types
model_types=("mixtral-8x7b-instruct-vllm")

# Loop through model types for full control set
for model_type in "${model_types[@]}"
do
    echo "Running model: $model_type"
    python print-insertions/eval-and-select.py model=$model_type data=humaneval-patch-011224-temp07-gpt4prints-exploded condition=print
done

