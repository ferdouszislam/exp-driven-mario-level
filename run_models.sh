#!/bin/bash
#SBATCH --account=def-drafiei
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --job-name=mario_vae_arr
#SBATCH --output=logs/%x-%A_%a.out

mkdir -p logs
module load python/3.11 cuda/12.2
source ~/ENV/bin/activate

cd ~/scratch/exp-driven-mario-level

SCRIPTS=(
    "model-training/initial_model_architecture/train.py --output-dir model-training/runs/m0_target --checkpoint-start-epoch 27 --checkpoint-end-epoch 37 --checkpoint-start-epoch 64 --checkpoint-end-epoch 74"
    "model-training/initial_model_architecture_softmax/train.py --output-dir model-training/runs/m0_softmax_target --checkpoint-start-epoch 2 --checkpoint-end-epoch 12 --checkpoint-start-epoch 18 --checkpoint-end-epoch 28 --checkpoint-start-epoch 74 --checkpoint-end-epoch 84"
    "model-training/initial_model_architecture_softmax_weighted/train.py --output-dir model-training/runs/m0_softmax_weighted_target --checkpoint-start-epoch 12 --checkpoint-end-epoch 22 --checkpoint-start-epoch 26 --checkpoint-end-epoch 36"
    "model-training/initial_model_architecture_weighted/train.py --output-dir model-training/runs/m0_weighted_target --checkpoint-start-epoch 9 --checkpoint-end-epoch 22 --checkpoint-start-epoch 86 --checkpoint-end-epoch 97"
    "model-training/model_arch_1/train.py --output-dir model-training/runs/m1_target --checkpoint-start-epoch 28 --checkpoint-end-epoch 49 --checkpoint-start-epoch 84 --checkpoint-end-epoch 98"
    "model-training/model_arch_1_softmax/train.py --output-dir model-training/runs/m1_softmax_target --checkpoint-start-epoch 27 --checkpoint-end-epoch 52"
    "model-training/model_arch_1_softmax_weighted/train.py --output-dir model-training/runs/m1_softmax_weighted_target --checkpoint-start-epoch 17 --checkpoint-end-epoch 27 --checkpoint-start-epoch 69 --checkpoint-end-epoch 89"
    "model-training/model_arch_1_weighted/train.py --output-dir model-training/runs/m1_weighted_target --checkpoint-start-epoch 75 --checkpoint-end-epoch 94"
    "model-training/model_arch_2/train.py --output-dir model-training/runs/m2_target --checkpoint-start-epoch 56 --checkpoint-end-epoch 66 --checkpoint-start-epoch 70 --checkpoint-end-epoch 89"
    "model-training/model_arch_2_softmax/train.py --output-dir model-training/runs/m2_softmax_target --checkpoint-start-epoch 41 --checkpoint-end-epoch 58 --checkpoint-start-epoch 75 --checkpoint-end-epoch 85"
    "model-training/model_arch_2_softmax_weighted/train.py --output-dir model-training/runs/m2_softmax_weighted_target --checkpoint-start-epoch 32 --checkpoint-end-epoch 42 --checkpoint-start-epoch 46 --checkpoint-end-epoch 66 --checkpoint-start-epoch 79 --checkpoint-end-epoch 89"
    "model-training/model_arch_2_weighted/train.py --output-dir model-training/runs/m2_weighted_target --checkpoint-start-epoch 5 --checkpoint-end-epoch 15 --checkpoint-start-epoch 26 --checkpoint-end-epoch 36 --checkpoint-start-epoch 72 --checkpoint-end-epoch 90"
)

CURRENT_SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
echo "Starting $CURRENT_SCRIPT at $(date)"
python -u $CURRENT_SCRIPT
echo "Finished $CURRENT_SCRIPT at $(date)"
