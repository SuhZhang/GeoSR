#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
#NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=4

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
OUTPUT_DIR="GeoSR3D"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="spar_234k,llava_hound_64k"                  # [DataArguments] Dataset with sampling rate

# ======================
# Vision Mask Configuration
# ======================
VISION_MASK_APPLY_PROB=0.5                        # Prob. to enable random visual token mask for a sample
VISION_MASK_PROB=0.8                              # Per-token random mask prob when enabled

validate_prob() {
    local name="$1"
    local value="$2"
    awk -v v="$value" 'BEGIN { exit !(v+0 == v && v >= 0 && v <= 1) }' || {
        echo "Invalid ${name}: ${value} (expected a number in [0, 1])"
        exit 1
    }
}

# Support positional usage:
#   bash scripts/train/train.sh 0.5 0.8 /path/to/output_dir
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
#   bash scripts/train/train.sh --vision-mask-apply-prob 0.5 --vision-mask-prob 0.8 --output-dir /path/to/output_dir
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
            echo "  bash scripts/train/train.sh [vision_mask_apply_prob] [vision_mask_prob] [output_dir]"
            echo "  bash scripts/train/train.sh --vision-mask-apply-prob 0.5 --vision-mask-prob 0.8 --output-dir /path/to/output_dir"
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
mkdir -p "$OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
LR=1e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to tensorboard \
            --use_geometry_encoder true \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method "gated" \
            --vision_mask_apply_prob $VISION_MASK_APPLY_PROB \
            --vision_mask_prob $VISION_MASK_PROB \
            > ${OUTPUT_DIR}/train.log 2>&1
