#!/bin/bash
#SBATCH --job-name=road_surface_train
#SBATCH --output=../logs/train_job_%j.out
#SBATCH --error=../logs/train_job_%j.err
#SBATCH --error=../logs/train_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu_4

# Load necessary modules
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/12.4

# Activate Conda environment
echo "Activating Conda environment..."
conda activate geoai
conda install conda=25.1.1
conda install numpy
conda install segmentation-models-pytorch

# Run the training script
echo "Starting training..."
python src/train_classifier.py --config config/street_surface.yaml

echo "Classification Training complete!"

# Run evaluation after training
echo "Starting evaluation..."
python src/evaluate.py --config config/street_surface.yaml --model-type best --task surface_classification
python src/evaluate.py --config config/street_surface.yaml --model-type final --task surface_classification

echo "Classification Evaluation complete!"
