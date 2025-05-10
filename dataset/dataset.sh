#!/bin/bash
#SBATCH --account=emalach_lab
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A/%a/logs.out
#SBATCH --job-name=preproc
#SBATCH --partition=sapphire

module load python/3.12.5-fasrc01
source niso/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directory for this job
#mkdir -p logs/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}

CLASSES=(alien ants armadillo bird1 bird2 camel cat centaur dino_ske dinosaur dog1 dog2 flamingo gorilla hand horse laptop man octopus pliers rabbit santa shark spiders woman)
TRANSFORMS=(elastic dropout noisy amputation)

COMBOS=()
for t in "${TRANSFORMS[@]}"; do
  for c in "${CLASSES[@]}"; do
    COMBOS+=("$t $c")
  done
done

read -r TRANSFORM CLASS <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "Running transform=$TRANSFORM, class=$CLASS"
python dataset/dataset.py --transform "$TRANSFORM" --cls "$CLASS"
