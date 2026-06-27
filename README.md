# OpenVLA LIBERO LoRA Demo

This repository is a robotics research portfolio project for reproducing OpenVLA inference evaluation on LIBERO manipulation tasks, then extending it with small-scale LoRA fine-tuning, action-sequence visualization, and experiment tracking.

## Current Status

- Phase 1 completed: repository setup and local environment check.
- Phase 2 completed: cloud setup scripts, official OpenVLA notes, LIBERO evaluation templates, LoRA templates, and action visualization utilities.
- Phase 3 completed: official checkpoint LIBERO-Spatial OSMesa baseline evaluation.
- Phase 4 completed at small scale: LoRA-50 training, checkpoint saving, merged model loading, and OSMesa evaluation loop.
- Current recommendation: pause larger training, inspect the LoRA-50 failure case, and tune conservatively before any longer run.

## Experiment Progress

- H800 cloud OpenVLA + LIBERO environment setup completed.
- Official `openvla-7b-finetuned-libero-spatial` checkpoint downloaded and loaded successfully on the cloud instance.
- EGL rendering was unstable for longer rollouts, with native aborts around 51 action steps.
- OSMesa rendering was verified as the stable evaluation path.
- Official checkpoint OSMesa baseline completed on `libero_spatial`:
  - `1 task x 3 trials`: `3/3` success.
  - `3 tasks x 3 trials`: `9/9` success.
  - No native abort, EGL, or `read_pixels` crash in the OSMesa baseline runs.
  - Rollout MP4 files were saved under `artifacts/rollouts/`.
- Current baseline log bundle: `artifacts/openvla_official_baseline_logs_latest.tar.gz`.
- LoRA-50 small training completed on top of the official LIBERO-Spatial checkpoint:
  - `max_steps=50`, `batch_size=4`, `lora_rank=32`, `learning_rate=5e-4`, `image_aug=True`.
  - Adapter saved remotely, about `463M`.
  - Merged HuggingFace model saved remotely, about `15G`.
  - `1 task x 1 trial`: `1/1` success, `83` action steps.
  - `3 tasks x 3 trials`: `8/9` success.
  - No OOM, CUDA crash, native abort, `read_pixels`, traceback, or core dump occurred.
- Current LoRA-50 log bundle: `artifacts/openvla_lora50_pipeline_logs_latest.tar.gz`.
- Latest rollout MP4 files are kept locally under `artifacts/rollouts/` and are intentionally ignored by Git.

## LoRA-50 Results

This LoRA experiment is a small workflow validation run on top of the official LIBERO-Spatial checkpoint. It is not a full fine-tuning reproduction from the base OpenVLA checkpoint.

| Model / run | Training setting | Evaluation setting | Result | Interpretation |
| --- | --- | --- | --- | --- |
| Official checkpoint baseline | Official `openvla-7b-finetuned-libero-spatial` | `libero_spatial`, OSMesa, `3 tasks x 3 trials` | `9/9` success | Reference baseline for this project |
| LoRA-50 | `max_steps=50`, `batch_size=4`, `lora_rank=32`, `learning_rate=5e-4`, `image_aug=True` | Same 3-task OSMesa subset | `8/9` success | Completed LoRA training/saving/loading/eval loop, but did not exceed the official baseline |

LoRA-50 per-task result:

| Task | Success rate | Action steps |
| --- | --- | --- |
| Pick up the black bowl between the plate and the ramekin and place it on the plate | `3/3` | 83, 74, 89 |
| Pick up the black bowl next to the ramekin and place it on the plate | `3/3` | 112, 115, 132 |
| Pick up the black bowl from table center and place it on the plate | `2/3` | 220, 115, 96 |

Episode 7 in the LoRA-50 `3 tasks x 3 trials` run is the failure case: task 3, trial 1, `success=False`, 220 action steps. This should be analyzed before increasing training length.

Do not report LoRA-50 as a performance improvement. The useful result is that the project now has an end-to-end LoRA workflow: RLDS data loading, LoRA adapter insertion, metric logging, adapter and merged-model saving, model loading, OSMesa rollout evaluation, log packaging, and video artifact tracking.

## Background

OpenVLA is a Vision-Language-Action model for robot control. This project is intended to:

- Reproduce OpenVLA inference evaluation on LIBERO manipulation tasks.
- Prepare a small, controlled LoRA fine-tuning workflow.
- Visualize predicted action sequences for analysis and presentation.
- Keep environment reports, experiment notes, and resume-ready summaries in one place.

## Project Phases

- Phase 1: repository setup and environment check
- Phase 2: OpenVLA environment setup
- Phase 3: LIBERO evaluation
- Phase 4: LoRA fine-tuning
- Phase 5: action visualization and resume packaging

## Local Environment Conclusion

The Phase 1 report detected an NVIDIA GeForce RTX 4060 Laptop GPU with about 8GB VRAM and CPU-only PyTorch in the local environment. This machine is not recommended for running OpenVLA-7B inference, LIBERO evaluation, or LoRA fine-tuning.

Use the local machine mainly for:

- Code editing and repository management.
- Reading official OpenVLA code.
- Documentation and experiment planning.
- Lightweight CSV action visualization.
- Git/GitHub workflow.

## Recommended Cloud GPU

- Recommended starting point: A100 40GB or higher.
- H800/A100 80GB is more suitable for OpenVLA LoRA experiments.
- On this cloud instance, use OSMesa for LIBERO evaluation:

```bash
RENDER_BACKEND=osmesa \
SAVE_VIDEO=1 \
MAX_TASKS=3 \
MAX_STEPS_PER_EPISODE=full \
CHECKPOINT=/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial \
NUM_TRIALS_PER_TASK=3 \
bash scripts/run_libero_eval.sh
```

## Environment Check

Run:

```bash
bash scripts/check_env.sh
```

The report will be written to:

```text
notes/environment_report.md
```

## Cloud Run Order

On the Linux cloud GPU instance:

```bash
git clone <THIS_REPO_URL>
cd openvla-libero-lora-demo
python scripts/check_env.py
bash scripts/setup_env.sh
RENDER_BACKEND=osmesa NUM_TRIALS_PER_TASK=1 bash scripts/run_libero_eval.sh
```

Then save the log path in:

```text
notes/experiment_log.md
```

Commit notes only after reviewing what was actually run. Do not commit checkpoints, Hugging Face cache, external repositories, or rollout videos.

## LIBERO Evaluation

The evaluation script defaults to the official LIBERO-Spatial checkpoint:

```bash
CHECKPOINT=openvla/openvla-7b-finetuned-libero-spatial
TASK_SUITE=libero_spatial
NUM_TRIALS_PER_TASK=1
CENTER_CROP=True
bash scripts/run_libero_eval.sh
```

The first real run may download the selected Hugging Face checkpoint. Do this only on a suitable Linux cloud GPU. For this project, the validated cloud path uses `RENDER_BACKEND=osmesa`.

## LoRA Template

The LoRA script is a conservative template and should not be used for long training until the dataset and dry-run plan are confirmed:

```bash
CONFIRM_TRAIN=YES \
DATA_ROOT_DIR=data/rlds \
DATASET_NAME=libero_spatial_no_noops \
bash scripts/run_lora_finetune.sh
```

Before training, confirm that the dataset exists and that the dataset name is registered in OpenVLA's RLDS dataloader.

## Action Visualization

Visualize a fake action sequence:

```bash
python scripts/visualize_actions.py
```

Or visualize a CSV with fields `step,dx,dy,dz,droll,dpitch,dyaw,gripper`:

```bash
python scripts/visualize_actions.py --input path/to/actions.csv
```

Outputs:

```text
results/figures/action_xyz_curve.png
results/figures/action_rotation_curve.png
results/figures/gripper_curve.png
```

## Safety Notes

Current repository scripts do not automatically download large models.

Current repository scripts do not automatically download large datasets.

Current repository scripts do not automatically run training.

Large downloads, LIBERO setup, OpenVLA checkpoints, and training commands should only be added after explicit confirmation.

Do not download BridgeData V2 at the start of this project.

Do not start with LoRA training.

Do not force OpenVLA-7B to run on the local 8GB GPU.

## TODO

- [x] Create repository skeleton.
- [x] Add repeatable environment check.
- [x] Confirm target Python version and package manager for cloud setup.
- [x] Add official OpenVLA setup command template.
- [x] Add LIBERO installation dependency template.
- [x] Add small debug evaluation command template.
- [x] Add LoRA fine-tuning command template with conservative defaults.
- [x] Add action visualization utilities.
- [x] Stabilize official checkpoint LIBERO evaluation with OSMesa.
- [x] Record official checkpoint OSMesa baseline.
- [ ] Inspect LoRA training entry point, dataset requirements, memory budget, and dry-run plan.
- [ ] Prepare resume-ready final package after LoRA or additional evaluation is actually completed.
