#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${BASE_MODEL:-openvla/openvla-7b}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-data/rlds}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/logs/lora_runs}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-results/logs/adapter_tmp}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
IMAGE_AUG="${IMAGE_AUG:-True}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_STEPS="${MAX_STEPS:-100}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-1000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_DIR="${LOG_DIR:-results/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/${LOG_DIR}/lora_finetune_${TIMESTAMP}.log"
OPENVLA_DIR="${PROJECT_ROOT}/external/openvla"

echo "=== OpenVLA LoRA fine-tuning template ==="
echo "This script is only a template until the dataset path and dataset registration are confirmed."
echo "Do not claim training has completed unless a real log exists and the experiment is recorded."
echo
echo "Memory guidance:"
echo "- 8GB VRAM: not suitable for OpenVLA-7B."
echo "- 24GB VRAM: high risk; only tiny experiments may work with aggressive memory controls."
echo "- 40GB+ VRAM: more reasonable for small LoRA experiments."
echo "- A100 80GB: closer to the official OpenVLA LoRA example."
echo
echo "Base model: ${BASE_MODEL}"
echo "Data root: ${DATA_ROOT_DIR}"
echo "Dataset name: ${DATASET_NAME}"
echo "Run root: ${RUN_ROOT_DIR}"
echo "Adapter temp dir: ${ADAPTER_TMP_DIR}"
echo "LoRA rank: ${LORA_RANK}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accumulation: ${GRAD_ACCUMULATION_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Image augmentation: ${IMAGE_AUG}"
echo "Save steps: ${SAVE_STEPS}"
echo "Max steps: ${MAX_STEPS}"
echo "Log file: ${LOG_FILE}"
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this fine-tuning template is intended for Linux cloud GPU instances."
  exit 1
fi

if [[ ! -d "${OPENVLA_DIR}" ]]; then
  echo "ERROR: external/openvla does not exist. Run scripts/setup_env.sh on the cloud instance first."
  exit 1
fi

if [[ ! -d "${PROJECT_ROOT}/${DATA_ROOT_DIR}" && ! -d "${DATA_ROOT_DIR}" ]]; then
  echo "ERROR: data directory not found: ${DATA_ROOT_DIR}"
  echo "Before training, confirm the dataset exists and that DATASET_NAME is registered in OpenVLA's RLDS dataloader."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Use a GPU instance with NVIDIA drivers installed."
  exit 1
fi
nvidia-smi

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"ERROR: failed to import torch: {exc}")
    sys.exit(1)
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: torch CUDA is not available.")
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "${PROJECT_ROOT}/${LOG_DIR}" "${PROJECT_ROOT}/${RUN_ROOT_DIR}" "${PROJECT_ROOT}/${ADAPTER_TMP_DIR}"

DATA_ROOT_ABS="$(cd "${PROJECT_ROOT}/${DATA_ROOT_DIR}" 2>/dev/null && pwd || cd "${DATA_ROOT_DIR}" && pwd)"
RUN_ROOT_ABS="${PROJECT_ROOT}/${RUN_ROOT_DIR}"
ADAPTER_TMP_ABS="${PROJECT_ROOT}/${ADAPTER_TMP_DIR}"

echo "About to prepare a LoRA command."
echo "Confirm before running:"
echo "- DATASET_NAME must already be registered in OpenVLA, usually in prismatic/vla/datasets/rlds/oxe/configs.py and transforms.py."
echo "- This command will download/load ${BASE_MODEL} if it is not cached."
echo "- This command starts training and may consume large GPU memory."
echo

COMMAND=(
  torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}"
  vla-scripts/finetune.py
  --vla_path "${BASE_MODEL}"
  --data_root_dir "${DATA_ROOT_ABS}"
  --dataset_name "${DATASET_NAME}"
  --run_root_dir "${RUN_ROOT_ABS}"
  --adapter_tmp_dir "${ADAPTER_TMP_ABS}"
  --lora_rank "${LORA_RANK}"
  --batch_size "${BATCH_SIZE}"
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --image_aug "${IMAGE_AUG}"
  --save_steps "${SAVE_STEPS}"
  --max_steps "${MAX_STEPS}"
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --run_id_note "small_debug"
)

printf 'Command template:\n'
printf ' %q' "${COMMAND[@]}"
printf '\n\n'

if [[ "${CONFIRM_TRAIN:-}" != "YES" ]]; then
  echo "Training not started. Set CONFIRM_TRAIN=YES to run this template after reviewing the command."
  exit 0
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
echo "CONFIRM_TRAIN=YES detected. Starting LoRA fine-tuning."
cd "${OPENVLA_DIR}"
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Fine-tuning command finished. Record the log path in notes/experiment_log.md:"
echo "${LOG_FILE}"
