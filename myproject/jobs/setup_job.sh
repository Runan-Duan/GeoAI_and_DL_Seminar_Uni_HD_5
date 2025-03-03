#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=../logs/setup_job_%j.out
#SBATCH --error=../logs/setup_job_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --gres gpu:1
#SBATCH --mem 10G

# Load necessary modules
module load miniconda

# Navigate to the project directory
cd ..

# Create Conda environment
echo "Creating Conda environment..."
conda env create -f environment.yml

# Activate the environment
echo "Activating Conda environment..."
conda activate geoai

# Create directories
echo "Creating directories..."
mkdir -p data/processed models logs results

# Link the dataset
echo "Linking dataset..."
ln -sf /test/StreetSurfaceVis_1024 data/raw

echo "Setup complete!"