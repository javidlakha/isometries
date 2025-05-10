#!/bin/bash
#SBATCH --account=kempner_emalach_lab
#SBATCH --array=0-1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=neural-isometries
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j/logs.out
#SBATCH --partition=kempner_h100
#SBATCH --time=2:00:00

echo "Starting job $SLURM_JOB_ID"

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

source ~/.bashrc
source niso/bin/activate
echo "Python interpreter: $(which python)"

export PYTHONPATH="$PWD:$PYTHONPATH"
echo $PYTHONPATH

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

ALPHA_VALUES=(12 24)
ALPHA=${ALPHA_VALUES[$SLURM_ARRAY_TASK_ID]}
export ALPHA
export SIGMA=6.0
export AE=0.5

echo "Using ALPHA=${ALPHA}"

JOB_DIR=logs/${SLURM_JOB_ID}/alpha${ALPHA}
mkdir -p $JOB_DIR

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > $JOB_DIR/gpu_usage.csv &
NVIDIA_SMI_PID=$!

set -e
cleanup() {
  rm -f "$JOB_DIR/data/dataset.mat"
  kill $NVIDIA_SMI_PID
}
trap cleanup EXIT

python experiments/mnist/deform/dataset/dataset.py --out $JOB_DIR/data --alpha $ALPHA --sigma $SIGMA
python experiments/mnist/deform/encode/train.py --out $JOB_DIR/ --alpha $ALPHA --sigma $SIGMA --ae $AE
python experiments/mnist/deform/predict/train.py \
  --in "$JOB_DIR/data/dataset.mat" \
  --weights "$JOB_DIR/0/checkpoints-0" \
  --out $JOB_DIR/
