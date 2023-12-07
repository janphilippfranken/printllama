#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPU
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning


# Change to the directory with your Python script
cd ~/research_projects/scai-tuning/experiments/cai

# Run
python generate_constitution.py