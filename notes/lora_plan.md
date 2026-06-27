# LoRA Plan

This note records the OpenVLA + LIBERO LoRA preparation state. It is a plan only. No LoRA training has been started.

## 1. Current Official Checkpoint Baseline

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Official checkpoint: `/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial`.
- Task suite: `libero_spatial`.
- Renderer: OSMesa.
- Baseline results:
  - `1 task x 3 trials`: `3/3` success.
  - `3 tasks x 3 trials`: `9/9` success.
  - No native abort, EGL, `read_pixels`, segmentation fault, or core dump in OSMesa baseline runs.
- Local baseline bundle: `artifacts/openvla_official_baseline_logs_latest.tar.gz`.
- Local rollout videos: `artifacts/rollouts/`.

This official checkpoint baseline is the reference point for later LoRA comparison.

## 2. LoRA Goal

The immediate goal is not long training. The next safe step is a tiny LoRA dry-run that verifies:

- RLDS/LIBERO data can be discovered and loaded.
- The OpenVLA model can be loaded without triggering unintended downloads.
- PEFT LoRA adapters can be inserted.
- One to ten optimizer steps can run.
- Adapter weights or a merged dry-run checkpoint can be saved.
- The saved model can later be evaluated with the already validated OSMesa LIBERO evaluation path.

## 3. Training Entry Candidates

Primary LoRA entry:

```text
external/openvla/vla-scripts/finetune.py
```

This is the official parameter-efficient fine-tuning script. It uses Hugging Face PEFT and LoRA.

Important config fields from `FinetuneConfig`:

| Field | Meaning | Current dry-run recommendation |
| --- | --- | --- |
| `vla_path` | Base OpenVLA model path or HF ID | Prefer a local path to avoid download. Current complete local model is the official LIBERO checkpoint. |
| `data_root_dir` | Root directory containing RLDS/TFDS datasets | `/root/autodl-tmp/openvla-libero-lora-demo/data/rlds` after data is prepared |
| `dataset_name` | Registered RLDS dataset name | `libero_spatial_no_noops` |
| `run_root_dir` | Output directory for logs and merged checkpoints | `results/lora_runs` |
| `adapter_tmp_dir` | Temporary LoRA adapter directory before merge | `results/lora_adapters_tmp` |
| `batch_size` | Per-process training batch size | Start with `1` |
| `grad_accumulation_steps` | Gradient accumulation | Start with `1` for dry-run, later `8-16` if needed |
| `max_steps` | Max gradient steps | `1-10` for dry-run |
| `save_steps` | Checkpoint interval | `1` only if accepting a merged checkpoint write; otherwise use a larger value for no-save smoke |
| `learning_rate` | LoRA learning rate | `5e-4` matches official script default |
| `image_aug` | Image augmentation | `True` for later real runs; `False` is acceptable for first data smoke |
| `lora_rank` | LoRA rank | `32` |
| `lora_dropout` | LoRA dropout | `0.0` |
| `use_quantization` | 4-bit quantized LoRA | Keep `False` on H800 80GB |

Secondary non-LoRA/full training entry:

```text
external/openvla/vla-scripts/train.py
```

This script supports resume fields such as `pretrained_checkpoint`, `is_resume`, `resume_step`, and `resume_epoch`, but it is not the preferred LoRA path. It targets larger FSDP-style VLA training.

Resume status:

- `vla-scripts/finetune.py` does not expose a clear resume flag for LoRA training.
- `vla-scripts/train.py` supports resume, but it is not the LoRA script.

Dry-run support:

- `vla-scripts/finetune.py` supports `max_steps` and `save_steps`, so it is suitable for a bounded dry-run.

## 4. Data Requirements

Current remote data state:

- `data/` exists but contains no demonstration files.
- `data/libero_datasets/datasets` exists but is empty.
- No LIBERO RLDS/TFDS training dataset is currently available in the project.

Registered LIBERO RLDS dataset names in OpenVLA:

- `libero_spatial_no_noops`
- `libero_object_no_noops`
- `libero_goal_no_noops`
- `libero_10_no_noops`

The finetuning script expects RLDS/TFDS-formatted datasets, not just rollout MP4 files.

Recommended next data step:

1. Prepare or download the official LIBERO demonstration dataset for `libero_spatial`.
2. Regenerate the no-op-filtered dataset if needed with:

```text
external/openvla/experiments/robot/libero/regenerate_libero_dataset.py
```

3. Convert or place the final RLDS/TFDS dataset so that OpenVLA can load:

```text
data/rlds/libero_spatial_no_noops
```

Do not download BridgeData V2 for this project stage.

Dataset size note:

- The exact LIBERO dataset size was not measured because the dataset is not present.
- Reserve at least tens of GB for raw LIBERO demonstrations plus processed/RLDS output.
- Do not start the download until the source, target directory, and expected disk usage are confirmed.

## 5. Environment And Memory Budget

Observed environment:

- GPU: NVIDIA H800 PCIe.
- VRAM: 81,559 MiB total, 0 MiB used during the check.
- PyTorch: `2.2.0+cu121`.
- CUDA available: `True`.
- Transformers: `4.40.1`.
- PEFT: `0.11.1`.
- Accelerate: `1.14.0`.
- NumPy: `1.26.4`.
- Disk on `/root/autodl-tmp`: about 136GB free at the time of inspection.

Memory recommendation:

- H800 80GB is suitable for OpenVLA LoRA.
- Official comments in `vla-scripts/finetune.py` state that one 80GB GPU can fit LoRA batch size 24 without quantization.
- For dry-run, use `batch_size=1`.
- For small real experiments, start conservatively with `batch_size=4-8`, then increase only after monitoring memory.
- Gradient accumulation is not required for a 1-step dry-run, but useful for later controlled effective batch sizes.
- Use BF16. H800 supports BF16 well.
- Keep `use_quantization=False` unless memory becomes a blocker.

Dependency risks:

- `wandb` import currently fails in the remote environment with a protobuf-related import error. Since `vla-scripts/finetune.py` imports `wandb` at module import time, this must be patched or fixed before running the dry-run.
- `flash_attn` is not installed. The LoRA script does not require it for the planned dry-run; do not install it unless a later performance need is confirmed.
- The OSMesa evaluation patch affects evaluation only. It does not change LoRA training.
- The evaluation SDPA/token patch affects evaluation utilities only. It does not change `vla-scripts/finetune.py`.

## 6. Dry-Run Command Draft

Current blocker: the LIBERO RLDS dataset is missing. Do not run this command until `data/rlds/libero_spatial_no_noops` exists and the `wandb` import issue is handled.

Recommended first mechanical dry-run after data exists:

```bash
cd /root/autodl-tmp/openvla-libero-lora-demo/external/openvla
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=disabled

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial \
  --data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs \
  --adapter_tmp_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --max_steps 2 \
  --save_steps 1 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --image_aug True \
  --shuffle_buffer_size 1000 \
  --save_latest_checkpoint_only True \
  --run_id_note libero_spatial_dryrun
```

Important interpretation:

- This command uses the already downloaded official LIBERO-Spatial checkpoint as `vla_path` to avoid downloading `openvla/openvla-7b`.
- That is useful for a mechanical dry-run, but it is not the same experimental question as training LoRA from the base `openvla/openvla-7b`.
- A true base-model LoRA comparison would require explicitly downloading `openvla/openvla-7b` later.
- With `save_steps=1`, the official script saves LoRA adapter weights and then merges them into a full HF model directory. This can write a large checkpoint directory even for a tiny dry-run.

Safer no-save data/backprop smoke variant:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial \
  --data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs \
  --adapter_tmp_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --max_steps 2 \
  --save_steps 1000 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --image_aug False \
  --shuffle_buffer_size 1000 \
  --run_id_note libero_spatial_no_save_smoke
```

This variant validates data loading, LoRA insertion, forward pass, backward pass, and optimizer steps, but it does not validate adapter saving.

## 7. Expected Output Paths

For the save-enabled dry-run, expected experiment ID shape:

```text
openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_dryrun--image_aug
```

Expected paths:

```text
results/lora_runs/<experiment_id>
results/lora_adapters_tmp/<experiment_id>
```

The official script saves:

- Dataset statistics into `run_root_dir/<experiment_id>`.
- Processor files into `run_root_dir/<experiment_id>`.
- Adapter weights into `adapter_tmp_dir/<experiment_id>`.
- A merged HF model into `run_root_dir/<experiment_id>` at checkpoint save time.

## 8. Evaluation After LoRA

If the official script creates a merged HF model in `results/lora_runs/<experiment_id>`, evaluate it with the existing OSMesa path:

```bash
cd /root/autodl-tmp/openvla-libero-lora-demo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

RENDER_BACKEND=osmesa \
SAVE_VIDEO=1 \
MAX_TASKS=3 \
MAX_STEPS_PER_EPISODE=full \
CHECKPOINT=/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/<experiment_id> \
NUM_TRIALS_PER_TASK=3 \
bash -x scripts/run_libero_eval.sh
```

If only adapter weights are saved, the current evaluation loader will need a small PEFT loading path that loads the base model and applies `PeftModel.from_pretrained(...)`.

## 9. Metrics To Record

Training dry-run metrics:

- exact command
- git commit
- dataset name and path
- `vla_path`
- LoRA rank/dropout
- batch size and gradient accumulation
- max steps and save steps
- GPU memory usage
- train loss
- action accuracy
- action L1 loss
- adapter or merged checkpoint path

Post-finetune evaluation metrics:

- task suite
- task count and trials per task
- success rate
- per-episode success
- action steps per episode
- rollout MP4 path
- native abort / EGL / `read_pixels` status

## 10. Risk Points

- No LIBERO training data is currently present.
- `wandb` import is currently broken and will stop `vla-scripts/finetune.py` before training starts unless patched or fixed.
- `openvla/openvla-7b` base model is not fully downloaded; only the official LIBERO-Spatial checkpoint is complete locally.
- A save-enabled dry-run can still write a large merged model directory.
- Official checkpoint LoRA-on-top is useful as a mechanical dry-run, but it is not equivalent to LoRA from the base OpenVLA model.
- Full training should not start until a tiny dry-run proves data loading, backpropagation, checkpoint saving, and OSMesa evaluation.

## 11. Do Not Commit

Do not commit:

- `external/`
- `data/`
- `hf_cache/`
- `openvla_checkpoints/`
- `runs/`
- `adapter-tmp/`
- `results/lora_runs/`
- `results/lora_adapters_tmp/`
- `*.safetensors`
- `*.bin`
- rollout MP4 files
- large log archives

## 12. Data Preparation Check

Date: 2026-06-27.

### Summary

- `libero_spatial_no_noops` is already registered in the OpenVLA dataloader.
- The current remote project does not contain any usable RLDS/TFDS training data.
- The official source for modified LIBERO RLDS data is the Hugging Face dataset repository:

```text
git@hf.co:datasets/openvla/modified_libero_rlds
```

- The OpenVLA README states that this repository contains modified LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets in RLDS format, about 10GB total.
- Do not download this dataset until the target path and expected disk use are explicitly confirmed.

### Registration Locations

`libero_spatial_no_noops` is registered in three places:

```text
external/openvla/prismatic/vla/datasets/rlds/oxe/configs.py
external/openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py
external/openvla/prismatic/vla/datasets/rlds/oxe/transforms.py
```

The relevant registered LIBERO dataset names are:

```text
libero_spatial_no_noops
libero_object_no_noops
libero_goal_no_noops
libero_10_no_noops
```

### Expected Data Layout

`vla-scripts/finetune.py` passes `data_root_dir` and `dataset_name` into `RLDSDataset`.

`RLDSDataset` uses the OXE materializer, which expects `data_root_dir` to be the base directory containing TFDS/RLDS dataset subdirectories. For this project, the recommended target is:

```text
/root/autodl-tmp/openvla-libero-lora-demo/data/rlds/libero_spatial_no_noops
```

The dry-run command should use:

```text
--data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds
--dataset_name libero_spatial_no_noops
```

### Raw Demonstrations And Regeneration

`external/openvla/experiments/robot/libero/regenerate_libero_dataset.py` regenerates raw LIBERO HDF5 demonstrations:

```text
input:  raw LIBERO HDF5 directory, such as ./LIBERO/libero/datasets/libero_spatial
output: no-op-filtered HDF5 directory, such as ./LIBERO/libero/datasets/libero_spatial_no_noops
```

The script notes that HDF5-to-RLDS conversion is not included there. The OpenVLA README points to `https://github.com/moojink/rlds_dataset_builder` for the conversion code. Therefore, for this project, the simplest path is to use the official preconverted `openvla/modified_libero_rlds` dataset rather than regenerating and converting from raw HDF5.

### Tiny Subset Feasibility

There is no confirmed official tiny-subset download path in the local OpenVLA code. A tiny subset is possible only if it preserves a valid TFDS/RLDS directory structure and statistics. MP4 rollout files and raw HDF5 files are not valid inputs to `vla-scripts/finetune.py`.

Practical recommendation:

- First download or prepare only the `libero_spatial_no_noops` RLDS data if the repository layout allows selective Git LFS pulling.
- Use `max_steps=2`, `batch_size=1`, and a small `shuffle_buffer_size` for the first dry-run.
- Do not use BridgeData V2 for the first LoRA dry-run.

### Current Remote Data State

Checked remote project:

```text
/root/autodl-tmp/openvla-libero-lora-demo/data
```

Current state:

- `data/` exists.
- `data/libero_datasets/datasets` exists but is empty.
- `data/rlds/libero_spatial_no_noops` does not exist.
- No files were found under `data/` at max depth 5.
- `du -sh data` reports 0.

Disk state:

```text
/root/autodl-tmp: 150GB total, 15GB used, 136GB available
```

The current disk is sufficient for the official modified LIBERO RLDS data as described by OpenVLA (~10GB total), plus small dry-run outputs. It is not a reason to download without confirmation.

### Wandb Patch Status

Remote file patched:

```text
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/vla-scripts/finetune.py
```

Patch behavior:

- Replaced top-level `import wandb` with a safe import.
- If `wandb` import fails, it prints a warning and sets `wandb = None`.
- Guarded `wandb.init(...)` and `wandb.log(...)` behind `wandb is not None`.
- This preserves `WANDB_MODE=disabled` compatibility and avoids failing before a dry-run starts.

Verification:

```text
python -m py_compile vla-scripts/finetune.py
```

Result: passed.

### Dry-Run Status

Current dry-run status:

```text
Not executable yet.
```

Reason:

```text
/root/autodl-tmp/openvla-libero-lora-demo/data/rlds/libero_spatial_no_noops
```

does not exist.

### No-Save Smoke Dry-Run Command Draft

Do not run until `libero_spatial_no_noops` exists.

```bash
cd /root/autodl-tmp/openvla-libero-lora-demo/external/openvla
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=disabled

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial \
  --data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs \
  --adapter_tmp_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --max_steps 2 \
  --save_steps 1000 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --image_aug False \
  --shuffle_buffer_size 1000 \
  --run_id_note libero_spatial_no_save_smoke
```

This command is intended to validate data loading, model loading, LoRA insertion, forward pass, backward pass, and optimizer steps without writing a large merged checkpoint.

### Save-Enabled Adapter Dry-Run Command Draft

Do not run until the no-save smoke run succeeds. This may write a large merged checkpoint directory.

```bash
cd /root/autodl-tmp/openvla-libero-lora-demo/external/openvla
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=disabled

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial \
  --data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs \
  --adapter_tmp_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --max_steps 2 \
  --save_steps 1 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --image_aug True \
  --shuffle_buffer_size 1000 \
  --save_latest_checkpoint_only True \
  --run_id_note libero_spatial_save_smoke
```

### Next Required Confirmation

Before any dry-run, confirm the data preparation action:

```text
Download or prepare official modified LIBERO RLDS data from openvla/modified_libero_rlds into:
/root/autodl-tmp/openvla-libero-lora-demo/data/rlds
```

Prefer starting with `libero_spatial_no_noops` only if selective Git LFS download is confirmed.

## 13. Spatial RLDS Data Prepared

Date: 2026-06-27.

The project prepared only the LIBERO-Spatial RLDS data needed for the first tiny LoRA dry-run. No LoRA training was started.

Download method:

- Remote did not have `git-lfs`, so Git LFS selective pull was not available.
- Used `huggingface_hub.snapshot_download` with:

```text
repo_id="openvla/modified_libero_rlds"
repo_type="dataset"
allow_patterns=["libero_spatial_no_noops/**"]
local_dir="/root/autodl-tmp/openvla-libero-lora-demo/data/rlds"
HF_ENDPOINT=https://hf-mirror.com
```

Downloaded target:

```text
/root/autodl-tmp/openvla-libero-lora-demo/data/rlds/libero_spatial_no_noops
```

Downloaded contents:

- `dataset_info.json`
- `features.json`
- `16` TFRecord shards:

```text
libero_spatial-train.tfrecord-00000-of-00016
...
libero_spatial-train.tfrecord-00015-of-00016
```

Size:

```text
data/rlds: about 1.8GB
```

TFDS metadata check:

```text
TFDS builder name: libero_spatial
TFDS version: 1.0.0
Splits: train
Train examples: 432
```

Selective download check:

```text
Non-spatial TFRecord count under data/rlds: 0
```

Disk after download:

```text
/root/autodl-tmp: about 134GB available
```

Next step:

- Run the no-save LoRA smoke dry-run only after explicit confirmation.
- First dry-run should use `max_steps=2`, `save_steps=1000`, `batch_size=1`, and `image_aug=False`.
- Do not run save-enabled adapter dry-run until the no-save smoke run succeeds.

## 14. No-Save LoRA Smoke Dry-Run

Date: 2026-06-27.

Purpose:

- Validate RLDS/TFDS data loading.
- Validate model loading from the local official LIBERO-Spatial checkpoint.
- Validate LoRA adapter insertion.
- Validate forward pass, backward pass, and optimizer update.
- Avoid saving LoRA adapters or merged model checkpoints.

Command summary:

```text
torchrun vla-scripts/finetune.py
--vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial
--data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds
--dataset_name libero_spatial_no_noops
--batch_size 1
--grad_accumulation_steps 1
--max_steps 2
--save_steps 1000
--learning_rate 5e-4
--lora_rank 32
--lora_dropout 0.0
--image_aug False
```

Result:

- Stage 1 no-save dry-run succeeded.
- Stage 1b metrics rerun succeeded after patching `finetune.py` to print metrics to stdout.
- Trainable parameters: `110,828,288 / 7,652,065,472 = 1.4483%`.
- The run reached the `max_steps=2` stop condition. The official script prints steps `0`, `1`, and `2` because the stop condition is checked after the optimizer update.
- No OOM occurred.
- The `wandb` protobuf import problem did not crash training because `wandb` is safely disabled.
- No `.safetensors` or `.bin` checkpoint was written in the no-save rerun.
- Only small `dataset_statistics.json` files were present under the no-save run directories.

Metrics from `results/logs/lora_no_save_metrics.log`:

```text
[TRAIN_METRICS] step=0 loss=0.000183 action_accuracy=1.000000 action_l1_loss=0.000000 lr=0.0005
[TRAIN_METRICS] step=1 loss=0.004093 action_accuracy=1.000000 action_l1_loss=0.000000 lr=0.0005
[TRAIN_METRICS] step=2 loss=0.603366 action_accuracy=0.857143 action_l1_loss=0.001120 lr=0.0005
```

Interpretation:

- This is a pipeline smoke test, not a performance claim.
- It confirms that data loading, model loading, LoRA insertion, backward pass, and optimizer update are connected.
- It does not produce an adapter or a model artifact for evaluation.

## 15. Save-Enabled LoRA Smoke Dry-Run

Date: 2026-06-27.

Purpose:

- Verify that the official fine-tuning script can save LoRA adapter weights.
- Verify whether the script also writes a merged HuggingFace model directory.
- Keep the run tiny with `max_steps=2`; do not treat this as real fine-tuning.

Command summary:

```text
torchrun vla-scripts/finetune.py
--vla_path /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial
--data_root_dir /root/autodl-tmp/openvla-libero-lora-demo/data/rlds
--dataset_name libero_spatial_no_noops
--run_root_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs
--adapter_tmp_dir /root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp
--batch_size 1
--grad_accumulation_steps 1
--max_steps 2
--save_steps 1
--learning_rate 5e-4
--lora_rank 32
--lora_dropout 0.0
--image_aug False
--save_latest_checkpoint_only True
--run_id_note libero_spatial_save_smoke
```

Result:

- Save-enabled dry-run succeeded.
- Trainable parameters: `110,828,288 / 7,652,065,472 = 1.4483%`.
- The script saved at Step 1 and Step 2 because `save_steps=1`; `save_latest_checkpoint_only=True` kept one output directory.
- Adapter was saved.
- A merged HuggingFace model directory was also saved.
- No OOM, CUDA crash, or wandb/protobuf crash occurred.

Metrics from `results/logs/lora_save_smoke.log`:

```text
[TRAIN_METRICS] step=0 loss=0.014726 action_accuracy=1.000000 action_l1_loss=0.000000 lr=0.0005
[TRAIN_METRICS] step=1 loss=0.007564 action_accuracy=1.000000 action_l1_loss=0.000000 lr=0.0005
[TRAIN_METRICS] step=2 loss=1.159093 action_accuracy=0.857143 action_l1_loss=0.085154 lr=0.0005
```

Outputs:

```text
Adapter:
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke
Size: 463M

Merged HF model:
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke
Size: 15G
```

Checkpoint files observed:

```text
adapter_config.json
adapter_model.safetensors
config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

Next step:

- Stage 3 should be a minimal OSMesa evaluation of the saved smoke model: `1 task x 1 trial`.
- Do not start 50-step LoRA training until the saved smoke model can be loaded and evaluated.
- Do not download or commit the `results/lora_runs` or `results/lora_adapters_tmp` directories.

## 16. Save-Smoke Model Evaluation

Date: 2026-06-27.

Purpose:

- Verify that the save-enabled LoRA smoke run produced a merged HuggingFace model that can be loaded by the LIBERO evaluation path.
- Run only `1 task x 1 trial` with OSMesa.
- Do not interpret this as a meaningful LoRA performance result, because the model came from a 2-step dry-run.

Merged model:

```text
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke
```

Adapter:

```text
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke
```

Lightweight load check:

- `config.json`: present.
- `model.safetensors.index.json`: present.
- Model shards: present.
- `dataset_statistics.json`: present.
- Tokenizer and processor files: present.
- `AutoConfig.from_pretrained(..., trust_remote_code=True)`: passed.
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`: passed.
- Config class: `OpenVLAConfig`.
- Processor class: `PrismaticProcessor`.

Evaluation settings:

```text
RENDER_BACKEND=osmesa
SAVE_VIDEO=1
MAX_TASKS=1
NUM_TRIALS_PER_TASK=1
MAX_STEPS_PER_EPISODE=full
CHECKPOINT=<merged model path>
```

Result:

- Stage 3 evaluation completed.
- The merged model loaded successfully.
- Evaluation entered `Task suite: libero_spatial`.
- Task: pick up the black bowl between the plate and the ramekin and place it on the plate.
- Episode success: `True`.
- Action steps: `74`.
- Rollout MP4 was saved.
- No native abort, `read_pixels` error, OOM, CUDA crash, segmentation fault, or core dump occurred.

Rollout MP4:

```text
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_30_21--episode=1--success=True--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4
```

Logs:

```text
/root/autodl-tmp/openvla-libero-lora-demo/results/logs/eval_lora_save_smoke_1task_1trial.log
/root/autodl-tmp/openvla-libero-lora-demo/results/logs/libero_eval_20260627_163002.log
```

Conclusion:

- Stage 3 passed: the saved smoke merged model can be loaded and evaluated with the stable OSMesa LIBERO path.
- Next recommended step is a controlled 50-step LoRA small training run, followed first by `1 task x 1 trial` evaluation before any wider comparison.

## 17. 50-Step LoRA Fine-Tuning

Date: 2026-06-27.

Scope:

- This is a small LoRA-on-official-LIBERO-checkpoint run.
- It is not a full OpenVLA fine-tuning reproduction from the base OpenVLA checkpoint.
- It is intended to validate the training and evaluation loop, not to claim performance improvement.

Training configuration:

```text
vla_path: /root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial
dataset_name: libero_spatial_no_noops
data_root_dir: /root/autodl-tmp/openvla-libero-lora-demo/data/rlds
max_steps: 50
save_steps: 50
batch_size: 4
grad_accumulation_steps: 1
learning_rate: 5e-4
lora_rank: 32
lora_dropout: 0.0
image_aug: True
save_latest_checkpoint_only: True
```

Training result:

- Stage 4 completed.
- The run reached `max_steps=50`.
- Trainable parameters: `110,828,288 / 7,652,065,472 = 1.4483%`.
- No OOM, CUDA crash, protobuf crash, or disk error occurred.
- Adapter was saved.
- A merged HuggingFace model was saved.

Final representative metrics from `results/logs/lora_spatial_50steps.log`:

```text
[TRAIN_METRICS] step=43 loss=0.696080 action_accuracy=0.928571 action_l1_loss=0.020448 lr=0.0005
[TRAIN_METRICS] step=44 loss=1.438865 action_accuracy=0.892857 action_l1_loss=0.010644 lr=0.0005
[TRAIN_METRICS] step=45 loss=1.086499 action_accuracy=0.857143 action_l1_loss=0.028011 lr=0.0005
[TRAIN_METRICS] step=46 loss=2.133811 action_accuracy=0.821429 action_l1_loss=0.018207 lr=0.0005
[TRAIN_METRICS] step=47 loss=0.245641 action_accuracy=0.964286 action_l1_loss=0.001120 lr=0.0005
[TRAIN_METRICS] step=48 loss=0.383422 action_accuracy=0.857143 action_l1_loss=0.006723 lr=0.0005
[TRAIN_METRICS] step=49 loss=0.518494 action_accuracy=0.892857 action_l1_loss=0.009244 lr=0.0005
[TRAIN_METRICS] step=50 loss=1.334518 action_accuracy=0.892857 action_l1_loss=0.000840 lr=0.0005
```

Outputs:

```text
Adapter:
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_lora50--image_aug
Size: 463M

Merged HF model:
/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_lora50--image_aug
Size: 15G
```

Lightweight load check:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)`: passed.
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`: passed.
- Config class: `OpenVLAConfig`.
- Processor class: `PrismaticProcessor`.

## 18. LoRA-50 OSMesa Evaluation

Date: 2026-06-27.

Evaluation settings:

```text
RENDER_BACKEND=osmesa
SAVE_VIDEO=1
MAX_STEPS_PER_EPISODE=full
CHECKPOINT=<LoRA-50 merged HF model>
```

### 1 task x 1 trial

- Result: completed.
- Success: `True`.
- Action steps: `83`.
- No native abort, `read_pixels` error, OOM, CUDA crash, segmentation fault, or core dump occurred.
- MP4:

```text
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_40_00--episode=1--success=True--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4
```

### 3 tasks x 3 trials

Overall result: `8/9` success.

| Task | Episodes | Success rate | Action steps |
| --- | --- | --- | --- |
| Pick up the black bowl between the plate and the ramekin and place it on the plate | 3 | `3/3` | 83, 74, 89 |
| Pick up the black bowl next to the ramekin and place it on the plate | 3 | `3/3` | 112, 115, 132 |
| Pick up the black bowl from table center and place it on the plate | 3 | `2/3` | 220, 115, 96 |

Per-episode outcome:

| Episode | Task | Success | Action steps |
| --- | --- | --- | --- |
| 1 | black bowl between plate and ramekin -> plate | True | 83 |
| 2 | black bowl between plate and ramekin -> plate | True | 74 |
| 3 | black bowl between plate and ramekin -> plate | True | 89 |
| 4 | black bowl next to ramekin -> plate | True | 112 |
| 5 | black bowl next to ramekin -> plate | True | 115 |
| 6 | black bowl next to ramekin -> plate | True | 132 |
| 7 | black bowl from table center -> plate | False | 220 |
| 8 | black bowl from table center -> plate | True | 115 |
| 9 | black bowl from table center -> plate | True | 96 |

Latest LoRA-50 rollout MP4 files:

```text
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_41_27--episode=9--success=True--task=pick_up_the_black_bowl_from_table_center_and_place.mp4
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_41_27--episode=8--success=True--task=pick_up_the_black_bowl_from_table_center_and_place.mp4
/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_41_27--episode=7--success=False--task=pick_up_the_black_bowl_from_table_center_and_place.mp4
```

Stability:

- No native abort.
- No `read_pixels` error.
- No OOM or CUDA crash.
- No Python traceback.

Comparison with official checkpoint baseline:

- Official checkpoint baseline on the same 3-task subset: `9/9`.
- LoRA-50 result on the same 3-task subset: `8/9`.
- This does not show performance improvement. The tiny 50-step LoRA-on-official-checkpoint run likely perturbed the already fine-tuned policy on one trial.
- The useful result is the completed training, checkpoint saving, OSMesa evaluation, logging, and rollout video loop.

Next recommendation:

- Pause larger training for now.
- Review the episode 7 failure video before launching new runs.
- Compare the failed rollout action sequence with successful task-3 rollouts.
- If continuing experiments, first try a lower learning rate, for example `1e-4` or `5e-5`, instead of directly increasing to 200 steps.
- Keep the same evaluation protocol so that any change is compared against both the official `9/9` baseline and the LoRA-50 `8/9` result.
