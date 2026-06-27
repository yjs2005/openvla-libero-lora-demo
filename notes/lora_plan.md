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
