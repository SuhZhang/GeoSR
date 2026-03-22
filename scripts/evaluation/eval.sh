#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# lmms_eval lives under src/, ensure the local package can be imported.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

benchmark="${BENCHMARK:-vsibench}" # choices: [vsibench, cvbench, blink_spatial]
output_path="${OUTPUT_PATH:-${PROJECT_ROOT}/logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")}"
default_model_path="${PROJECT_ROOT}/outputs/geosr3d_train"
model_path="${MODEL_PATH:-${default_model_path}}"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: neither python nor python3 was found in PATH." >&2
    exit 1
fi

if [[ -d "${model_path}" ]]; then
    model_path="$(cd -- "${model_path}" && pwd)"
    if [[ ! -f "${model_path}/config.json" ]]; then
        echo "Error: missing config.json under model_path: ${model_path}" >&2
        exit 1
    fi
elif [[ "${model_path}" = /* || "${model_path}" = ./* || "${model_path}" = ../* ]]; then
    echo "Error: model_path does not exist: ${model_path}" >&2
    echo "Hint: set MODEL_PATH to a valid local checkpoint directory or a Hugging Face repo id." >&2
    exit 1
fi

echo "Using model_path: ${model_path}"
echo "Running benchmark: ${benchmark}"
echo "Output path: ${output_path}"

"${PYTHON_BIN}" -m lmms_eval \
    --model geosr3d \
    --model_args pretrained="${model_path}",use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks "${benchmark}" \
    --batch_size 1 \
    --output_path "${output_path}"
