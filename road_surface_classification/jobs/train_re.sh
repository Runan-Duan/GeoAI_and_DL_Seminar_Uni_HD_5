#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time 30-00:00:00
#SBATCH -p octane 
#SBATCH -o resnet50.out
#SBATCH -e resnet50.err


module load anaconda/3 
eval "$(conda shell.bash hook)"
source ~/conda_envs/seml_pyt-tf/bin/activate
echo "Active Conda env: $(conda env list | grep '*')"


echo "running train.py"
python train.py \
    --model_name resnet50 \
    --models_dir './models'\
    --lr 0.001 \
    --epochs 120 \
    --batch_size 32 \
    --logs_dir './logs'\
    --data_path 'data/raw/StreetSurfaceVis_1024/'


echo "done"