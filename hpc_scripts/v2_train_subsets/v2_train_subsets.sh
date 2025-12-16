VAL_FILE="massivekb_data/massiveKB_3cac0386/subsets/val.mgf"
CONFIG_FILE="hpc_scripts/v2_train_subsets/default.yaml"

for NUM_SPECTRA in 20; do
  for NUM_PEP in 4243254; do
    TRAIN_FILE="massivekb_data/massiveKB_3cac0386/subsets/train_${NUM_SPECTRA}s_${NUM_PEP}p.mgf"
    OUTPUT_DIR="logs/v2_train_subsets/${NUM_SPECTRA}s_${NUM_PEP}p/"
    PBS_LOG_FILE="logs/v2_train_subsets/${NUM_SPECTRA}s_${NUM_PEP}p/pbs_$(date +%y%m%d%H%M%S%4N).out"
    qsub -o $PBS_LOG_FILE -j oe hpc_scripts/v2_train_subsets/v2_train_subsets.pbs -v TRAIN_FILE=$TRAIN_FILE,VAL_FILE=$VAL_FILE,OUTPUT_DIR=$OUTPUT_DIR,CONFIG_FILE=$CONFIG_FILE
    echo -o $PBS_LOG_FILE -j oe hpc_scripts/v2_train_subsets/v2_train_subsets.pbs -v TRAIN_FILE=$TRAIN_FILE,VAL_FILE=$VAL_FILE,OUTPUT_DIR=$OUTPUT_DIR,CONFIG_FILE=$CONFIG_FILE
  done
done
