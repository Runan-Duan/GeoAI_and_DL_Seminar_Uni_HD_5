#!/bin/bash
#SBATCH --job-name=road_surface_train
#SBATCH --output=../logs/evaluate_job_%j.out
#SBATCH --error=../logs/evaluate_job_%j.err
#SBATCH --error=../logs/evaluate_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu_4

# Load necessary modules
module load devel/miniconda/23.9.0-py3.9.15

# Activate Conda environment
echo "Activating Conda environment..."
conda activate geoai

# Run evaluation after training
echo "Starting evaluation..."
python evaluate.py --config config/street_surface.yaml --model-type best --task classification
python evaluate.py --config config/street_surface.yaml --model-type final --task classification

echo "Evaluation complete!"
