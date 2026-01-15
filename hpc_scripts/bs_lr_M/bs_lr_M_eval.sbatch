#!/bin/sh
#SBATCH -A 2025_048
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

echo "Starting job $CONFIG_FILE at $(date)"

module purge

cd $SLURM_SUBMIT_DIR

eval "$(/readonly${VSC_SCRATCH_PROJECTS_BASE}/2025_048/miniconda3/bin/conda shell.bash hook)"
conda activate casanovo-scaling-readonly

READONLY_BASE="/readonly$VSC_SCRATCH_PROJECTS_BASE/2025_048/casanovo-scaling"

echo "Sequencing validation data for $CONFIG_FILE at $(date)"

/readonly$VSC_SCRATCH_PROJECTS_BASE/2025_048/miniconda3/envs/casanovo-scaling-readonly/bin/casanovo sequence \
$READONLY_BASE/massivekb_data/massiveKB_3cac0386/subsets/val.mgf \
--config $CONFIG_FILE \
--model $OUTPUT_DIR/train.best.ckpt \
-d $OUTPUT_DIR \
-o validation \
-f
