#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=/home/${USER}/repos/sapiens

#----------------------------set your data and output directories----------------------------------------------
DATA_ROOT="${BEDLAM2_DATA_ROOT:-/media/s/SF_backup/bedlam2}"
SEQ_PATHS_FILE='data/bedlam2_splits/test_seqs.txt'

#--------------------------MODEL CARD---------------
MODEL='sapiens_0.3b-50e_bedlam2-640x384'

# Update the timestamp below to match your training run directory:
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/Outputs/train/bedlam2/${MODEL}/node/03-15-2026_18:14:31/best_bedlam_mpjpe_body_epoch_*.pth"

CONFIG_FILE="configs/sapiens_pose/bedlam2/${MODEL}.py"
OUTPUT="Outputs/demo/bedlam2/${MODEL}"

#---------------------------INFERENCE PARAMS--------------------------------------------------
NUM_SAMPLES=200
BATCH_SIZE=8

##-------------------------------------inference-------------------------------------
export PYTHONPATH="${SAPIENS_CHECKPOINT_ROOT}/pose:${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python demo/demo_bedlam2.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --data-root ${DATA_ROOT} \
    --seq-paths-file ${SEQ_PATHS_FILE} \
    --output-root ${OUTPUT} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --device cuda:0

# Go back to the original script's directory
cd - || exit

echo "Done. Results saved to ${OUTPUT}"
