#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPU
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate printllama

# Set CUDA visible devices to use both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Change to the directory with your Python script
cd ~/research_projects/printllama/experiments/apps

# Run
torchrun --nproc_per_node=1 add_codellama_prints.py