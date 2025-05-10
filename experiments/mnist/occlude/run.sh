#!/bin/bash
#SBATCH --account=kempner_emalach_lab
#SBATCH --array=0-4
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=neural-isometries
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j/logs.out
#SBATCH --partition=kempner_h100
#SBATCH --time=3:00:00
#SBATCH --exclude=holygpu8a15102

echo "Starting job $SLURM_JOB_ID"

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

source ~/.bashrc
source niso/bin/activate
echo "Python interpreter: $(which python)"

export PYTHONPATH="$PWD:$PYTHONPATH"
echo $PYTHONPATH

BOX_FRAC_VALUES=(0.1 0.2 0.3 0.5 0.6)
BOX_FRAC=${BOX_FRAC_VALUES[$SLURM_ARRAY_TASK_ID]}
export BOX_FRAC
echo "Using BOX_FRAC=${BOX_FRAC}"

JOB_DIR=logs/${SLURM_JOB_ID}/box${BOX_FRAC}
mkdir -p $JOB_DIR

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > $JOB_DIR/gpu_usage.csv &
NVIDIA_SMI_PID=$!

set -e
cleanup() {
  rm -f "$JOB_DIR/data/dataset.mat"
  kill $NVIDIA_SMI_PID
}
trap cleanup EXIT

python experiments/mnist/occlude_rotate/dataset/dataset.py --out $JOB_DIR/data --box-frac $BOX_FRAC
python experiments/mnist/occlude_rotate/encode/train.py --out $JOB_DIR/ --box-frac $BOX_FRAC
python experiments/mnist/occlude_rotate/predict/train.py \
  --in "$JOB_DIR/data/dataset.mat" \
  --weights "$JOB_DIR/0/checkpoints-0" \
  --out $JOB_DIR/
