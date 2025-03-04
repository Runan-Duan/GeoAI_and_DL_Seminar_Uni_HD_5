#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=../logs/setup_job_%j.out
#SBATCH --error=../logs/setup_job_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --partition=gpu_4

# Load necessary modules
module load devel/miniconda/23.9.0-py3.9.15

# Create Conda environment
echo "Creating Conda environment..."
conda env create -f environment.yml

# Activate the environment
echo "Activating Conda environment..."
conda activate geoai

# Create directories
echo "Creating directories..."
mkdir -p data/processed models logs results


echo "Setup complete!"