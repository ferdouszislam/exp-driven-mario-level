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
    "model-training/initial_model_architecture/train.py --output-dir model-training/runs/m0"
    "model-training/initial_model_architecture_weighted/train.py --output-dir model-training/runs/m0_weighted"
    "model-training/initial_model_architecture_softmax/train.py --output-dir model-training/runs/m0_softmax"
    "model-training/initial_model_architecture_softmax_weighted/train.py --output-dir model-training/runs/m0_softmax_weighted"
    "model-training/model_arch_1/train.py --output-dir model-training/runs/m1"
    "model-training/model_arch_1_weighted/train.py --output-dir model-training/runs/m1_weighted"
    "model-training/model_arch_1_softmax/train.py --output-dir model-training/runs/m1_softmax"
    "model-training/model_arch_1_softmax_weighted/train.py --output-dir model-training/runs/m1_softmax_weighted"
    "model-training/model_arch_2/train.py --output-dir model-training/runs/m2"
    "model-training/model_arch_2_weighted/train.py --output-dir model-training/runs/m2_weighted"
    "model-training/model_arch_2_softmax/train.py --output-dir model-training/runs/m2_softmax"
    "model-training/model_arch_2_softmax_weighted/train.py --output-dir model-training/runs/m2_softmax_weighted"
)

CURRENT_SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
echo "Starting $CURRENT_SCRIPT at $(date)"
python -u $CURRENT_SCRIPT
echo "Finished $CURRENT_SCRIPT at $(date)"
