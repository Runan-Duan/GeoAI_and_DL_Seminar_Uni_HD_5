#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time 30-00:00:00
#SBATCH -p octane 
#SBATCH -o convnext_small.out
#SBATCH -e convnext_small.err


module load anaconda/3 
eval "$(conda shell.bash hook)"
source ~/conda_envs/seml_pyt-tf/bin/activate
echo "Active Conda env: $(conda env list | grep '*')"


echo "running test.py"
python src/test.py \
    --model_name convnext_small \
    --models_dir './models'\
    --batch_size 128 \
    --data_path 'data/raw/StreetSurfaceVis_1024/'

echo "done"