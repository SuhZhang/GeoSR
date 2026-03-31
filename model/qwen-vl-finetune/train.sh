#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GEOSR4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEOSR4D_DATA_ROOT="${GEOSR4D_DATA_ROOT:-$GEOSR4D_ROOT/data}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
#NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=8
#NPROC_PER_NODE=8
# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="${GEOSR4D_OUTPUT_DIR:-$GEOSR4D_ROOT/outputs/geosr4d_train}"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models
PI3_PATH="${GEOSR4D_PI3_PATH:-$GEOSR4D_DATA_ROOT/models/Pi3_hf/model.safetensors}"

export GEOSR4D_TRAIN_QA_PATH="${GEOSR4D_TRAIN_QA_PATH:-$GEOSR4D_DATA_ROOT/spatial_reasoning/train_qas.json}"
export GEOSR4D_TRAIN_VIDEO_ROOT="${GEOSR4D_TRAIN_VIDEO_ROOT:-$GEOSR4D_DATA_ROOT/spatial_reasoning/videos_train}"

# ======================
# Model Configuration
# ======================
DATASETS="spatial_reasoning%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Vision Mask Configuration
# ======================
VISION_MASK_APPLY_PROB=0.5                        # Prob. to enable visual zero-mask for a sample
VISION_MASK_PROB=0.8                              # Per-token zero-mask prob when enabled

validate_prob() {
    local name="$1"
    local value="$2"
    awk -v v="$value" 'BEGIN { exit !(v+0 == v && v >= 0 && v <= 1) }' || {
        echo "Invalid ${name}: ${value} (expected a number in [0, 1])"
        exit 1
    }
}

# Support positional usage:
#   bash train.sh 0.5 0.8 /path/to/output_dir
if [[ $# -ge 1 && "$1" != --* ]]; then
    VISION_MASK_APPLY_PROB="$1"
    shift
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
    VISION_MASK_PROB="$1"
    shift
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

# Support named usage:
#   bash train.sh --vision-mask-apply-prob 0.5 --vision-mask-prob 0.8 --output-dir /path/to/output_dir
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vision-mask-apply-prob|--vision_mask_apply_prob)
            VISION_MASK_APPLY_PROB="$2"
            shift 2
            ;;
        --vision-mask-prob|--vision_mask_prob)
            VISION_MASK_PROB="$2"
            shift 2
            ;;
        --output-dir|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage:"
            echo "  bash train.sh [vision_mask_apply_prob] [vision_mask_prob] [output_dir]"
            echo "  bash train.sh --vision-mask-apply-prob 0.5 --vision-mask-prob 0.8 --output-dir /path/to/output_dir"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help to see supported arguments."
            exit 1
            ;;
    esac
done

validate_prob "VISION_MASK_APPLY_PROB" "$VISION_MASK_APPLY_PROB"
validate_prob "VISION_MASK_PROB" "$VISION_MASK_PROB"
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Invalid OUTPUT_DIR: empty value"
    exit 1
fi
echo "Using vision mask apply prob: $VISION_MASK_APPLY_PROB"
echo "Using vision mask prob: $VISION_MASK_PROB"
echo "Using output dir: $OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --pi3_path $PI3_PATH \
         --vision_mask_apply_prob $VISION_MASK_APPLY_PROB \
         --vision_mask_prob $VISION_MASK_PROB \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-6 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 8192 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 230400 \
         --min_pixels 784 \
         --base_interval 1 \
         --video_max_frames 32 \
         --video_min_frames 32 \
         --video_max_frame_pixels 230400 \
         --video_min_frame_pixels 784 \
         --num_train_epochs 1 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 100 \
         --save_total_limit 1 \
         --max_grad_norm 1 \
         --report_to tensorboard \
         --deepspeed ./scripts/zero3_offload.json \



# datasets

# bash train.sh --vision-mask-prob 0.8 --vision-mask-apply-prob 0.5 --output-dir ./outputs/geosr4d_train
