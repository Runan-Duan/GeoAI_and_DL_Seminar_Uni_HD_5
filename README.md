# StreetSurfaceVis Road Surface Classification

This project classifies road surface types using the StreetSurfaceVis dataset.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd project
```

2. Set up the Conda environment:
```bash
conda env create -f environment.yml
conda activate geoai
```

3. Link the dataset:
```bash
ln -s /test/StreetSurfaceVis_1024 data/raw
```

4. Run the training script:
```bash
sbatch jobs/train_job.sh
```


## Directory Structure
data/raw/: Raw dataset (linked to /test/StreetSurfaceVis_1024).

data/processed/: Processed data (e.g., train/val/test splits).

code/: Source code and scripts.

jobs/: Job submission scripts for the cluster.

models/: Saved models.

logs/: Training logs.

results/: Results (e.g., predictions, visualizations).

## License
This project is licensed under the MIT License.