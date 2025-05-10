#!/bin/bash
#SBATCH --account=kempner_emalach_lab
#SBATCH --array=1-1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=neural-isometries
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j/logs.out
#SBATCH --partition=kempner_h100
#SBATCH --time=24:00:00

echo "Starting job $SLURM_JOB_ID"

echo "amputation"

# Load modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

# Load API keys
source ~/.bashrc

# Activate virtual environment
source niso/bin/activate
echo "Python interpreter: $(which python)"

# Add current directory to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
echo $PYTHONPATH

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

# Log GPU usage in background
mkdir -p logs/$SLURM_JOB_ID
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > logs/$SLURM_JOB_ID/gpu_usage.csv &
NVIDIA_SMI_PID=$!

set -e
cleanup() {
  rm -f "logs/$SLURM_JOB_ID/data/dataset.mat"
  kill $NVIDIA_SMI_PID
}
trap cleanup EXIT

export DATA_DIR="data/shrec11_aug/processed_amputation/"

python3 experiments/shrec11/amputation/combine.py --data $DATA_DIR
python3 experiments/shrec11/amputation/encode/train.py --in $DATA_DIR --out "logs/$SLURM_JOB_ID/"
python3 experiments/shrec11/amputation/predict/train.py \
  --in $DATA_DIR \
  --weights "logs/$SLURM_JOB_ID/0/checkpoints-0" \
  --out logs/$SLURM_JOB_ID/
