cd ../../../..

###--------------------------------------------------------------
DEVICES=0,
# DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

##---------copy this to the slurm script-----------------
####-----------------MODEL_CARD----------------------------
DATASET='bedlam2'
MODEL="sapiens_0.3b-50e_${DATASET}-640x384"
JOB_NAME="pose_$MODEL"
TRAIN_BATCH_SIZE_PER_GPU=2

RESUME_FROM=''
LOAD_FROM=''

# Data root: set BEDLAM2_DATA_ROOT env var to override, e.g.:
#   BEDLAM2_DATA_ROOT=/data/bedlam2 bash node.sh
DATA_ROOT="${BEDLAM2_DATA_ROOT:-}"

##---------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
CONFIG_FILE=configs/sapiens_pose/${DATASET}/${MODEL}.py
OUTPUT_DIR="Outputs/train/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

###--------------------------------------------------------------
DATA_ROOT_OPT=""
[ -n "$DATA_ROOT" ] && DATA_ROOT_OPT="data_root=$DATA_ROOT"

if [ -n "$LOAD_FROM" ]; then
    OPTIONS="train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU load_from=$LOAD_FROM $DATA_ROOT_OPT"
else
    OPTIONS="train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU $DATA_ROOT_OPT"
fi

if [ -n "$RESUME_FROM" ]; then
    CMD_RESUME="--resume ${RESUME_FROM}"
else
    CMD_RESUME=""
fi

export TF_CPP_MIN_LOG_LEVEL=2

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TRAIN_BATCH_SIZE_PER_GPU=2

    OPTIONS="$(echo "train_dataloader.batch_size=${TRAIN_BATCH_SIZE_PER_GPU} train_dataloader.num_workers=0 train_dataloader.persistent_workers=False")"
    # OPTIONS="$(echo "train_dataloader.batch_size=${TRAIN_BATCH_SIZE_PER_GPU}")" ## faster debug

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} --no-validate --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    SEED='0'

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            --seed ${SEED} \
            --cfg-options ${OPTIONS} \
            ${CMD_RESUME} \
            | tee ${LOG_FILE}

fi
