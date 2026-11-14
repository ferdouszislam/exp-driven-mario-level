#!/bin/bash
#SBATCH --account=def-drafiei_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --job-name=mario_vae_arr
#SBATCH --output=logs/%x-%A_%a.out

mkdir -p logs
module load python/3.11 cuda/12.2
source ~/ENV/bin/activate

cd ~/scratch/exp-driven-mario-level

SCRIPTS=(
    "model-training/initial_model_architecture_softmax/train.py --output-dir model-training/runs/m0_softmax_target --checkpoint-start-epoch 1 --checkpoint-end-epoch 100"
    "model-training/initial_model_architecture_softmax_weighted/train.py --output-dir model-training/runs/m0_softmax_weighted_target --checkpoint-start-epoch 1 --checkpoint-end-epoch 100"
    "model-training/model_arch_2_softmax_weighted/train.py --output-dir model-training/runs/m2_softmax_weighted_target --checkpoint-start-epoch 1 --checkpoint-end-epoch 100"
)

CURRENT_SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
echo "Starting $CURRENT_SCRIPT at $(date)"
python -u $CURRENT_SCRIPT
echo "Finished $CURRENT_SCRIPT at $(date)"
