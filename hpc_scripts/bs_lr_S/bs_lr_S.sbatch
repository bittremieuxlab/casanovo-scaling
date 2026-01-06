#SBATCH -A 2025_048
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

echo "Starting job $CONFIG_FILE at $(date)"

cd $SLURM_SUBMIT_DIR

eval "$(/readonly${VSC_SCRATCH_PROJECTS_BASE}/2025_048/miniconda3/bin/conda shell.bash hook)"
conda activate casanovo-scaling-readonly

READONLY_BASE="/readonly$VSC_SCRATCH_PROJECTS_BASE/2025_048/casanovo-scaling"

echo "Determining batch size for $CONFIG_FILE at $(date)"

/readonly$VSC_SCRATCH_PROJECTS_BASE/2025_048/miniconda3/envs/casanovo-scaling-readonly/bin/casanovo determine-max-batch-size \
$READONLY_BASE/massivekb_data/scaling_data_max_100000/bs_finder.mgf \
--config $CONFIG_FILE \
-d $OUTPUT_DIR \
-o bs_finder \
-f

echo "Training $CONFIG_FILE at $(date)"

/readonly$VSC_SCRATCH_PROJECTS_BASE/2025_048/miniconda3/envs/casanovo-scaling-readonly/bin/casanovo train \
$READONLY_BASE/$TRAIN_FILE \
--validation_peak_path $READONLY_BASE/$VAL_FILE \
--config $BS_CONFIG_FILE \
-d $OUTPUT_DIR \
-o train \
-f

#casanovo sequence -e \
#$VAL_FILE \
#--config casanovo_train_subsets_sequence.yaml \
#--model ${OUTPUT_DIR}train.best.ckpt \
#-d $OUTPUT_DIR \
#-o validation \
#-f
#
#casanovo sequence -e \
#$TEST_FILE \
#--config casanovo_train_subsets_sequence.yaml \
#--model ${OUTPUT_DIR}train.best.ckpt \
#-d $OUTPUT_DIR \
#-o test \
#-f