#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
CENTER_CROP="${CENTER_CROP:-True}"
LOG_DIR="${LOG_DIR:-results/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/${LOG_DIR}/libero_eval_${TIMESTAMP}.log"
OPENVLA_DIR="${PROJECT_ROOT}/external/openvla"

echo "=== OpenVLA LIBERO smoke-test evaluation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Task suite: ${TASK_SUITE}"
echo "Trials per task: ${NUM_TRIALS_PER_TASK}"
echo "Center crop: ${CENTER_CROP}"
echo "Log file: ${LOG_FILE}"
echo
echo "Important:"
echo "- First run will download the Hugging Face checkpoint."
echo "- This requires a Linux cloud GPU instance."
echo "- Do not run on the local 8GB RTX 4060 Laptop GPU."
echo "- Keep NUM_TRIALS_PER_TASK=1 for the first smoke test."
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this evaluation script is intended for Linux cloud GPU instances."
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/README.md" || ! -d "${PROJECT_ROOT}/scripts" ]]; then
  echo "ERROR: could not resolve project root correctly. Run from this repository or keep the script in scripts/."
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

if [[ ! -d "${OPENVLA_DIR}" ]]; then
  echo "ERROR: external/openvla does not exist. Run scripts/setup_env.sh on the cloud instance first."
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/${LOG_DIR}"

echo "Starting official LIBERO smoke test. This may download a large checkpoint on first run."
cd "${OPENVLA_DIR}"
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name "${TASK_SUITE}" \
  --center_crop "${CENTER_CROP}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --local_log_dir "${PROJECT_ROOT}/${LOG_DIR}/openvla_libero_internal" \
  --use_wandb False \
  2>&1 | tee "${LOG_FILE}"

echo "Evaluation command finished. Save the log path in notes/experiment_log.md:"
echo "${LOG_FILE}"
