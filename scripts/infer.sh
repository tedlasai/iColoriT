export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}

# path to model and validation dataset
MODEL_PATH='/home/tedlasai/iColoriT/icolorit_base_4ch_patch16_224.pth'
VAL_DATA_PATH='/home/tedlasai/iColoriT/imagenet_val_begin'
VAL_HINT_DIR='/home/tedlasai/iColoriT/h2'
# Set the path to save checkpoints
PRED_DIR='/home/tedlasai/iColoriT/checkpoints'

# other options
opt=${2:-}

# batch_size can be adjusted according to the graphics card
python infer.py \
    --model_path=${MODEL_PATH} \
    --val_data_path=${VAL_DATA_PATH} \
    --val_hint_dir=${VAL_HINT_DIR} \
    --pred_dir=${PRED_DIR} \
    $opt