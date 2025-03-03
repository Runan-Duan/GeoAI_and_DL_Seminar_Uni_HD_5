#!/bin/bash
#SBATCH --job-name=road_surface_train
#SBATCH --output=../logs/train_job_%j.out
#SBATCH --error=../logs/train_job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G

# Load necessary modules
module load miniconda
# devel/miniconda/23.9.0-py3.9.15
# Activate Conda environment
echo "Activating Conda environment..."
conda activate geoai

# Navigate to the code directory
cd ../code

# Run the training script
echo "Starting training..."
python scripts/train.py --config ../config/street_surface.yaml

# Optionally run evaluation after training
echo "Starting evaluation..."
python scripts/evaluate.py --config ../config/street_surface.yaml --model-path ../models/road_surface_classification.pth

echo "Training and evaluation complete!"