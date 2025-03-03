#!/bin/bash
#SBATCH --job-name=road_surface_evaluate
#SBATCH --output=../logs/evaluate_job_%j.out
#SBATCH --error=../logs/evaluate_job_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=8G

# Load necessary modules
module load miniconda

# Activate Conda environment
echo "Activating Conda environment..."
conda activate geoai

# Navigate to the code directory
cd ../code

# Run the evaluation script
echo "Starting evaluation..."
python scripts/evaluate.py --config ../config/street_surface.yaml --model-path ../models/road_surface_classification.pth

echo "Evaluation complete!"