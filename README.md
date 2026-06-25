# OpenVLA LIBERO LoRA Demo

This repository is a robotics research portfolio project for reproducing OpenVLA inference evaluation on LIBERO manipulation tasks, then extending it with small-scale LoRA fine-tuning, action-sequence visualization, and experiment tracking.

## Current Status

- Phase 1 completed: repository setup and local environment check.
- Phase 2 in progress: official OpenVLA repository reading notes, Linux cloud GPU setup script, LIBERO smoke-test script, LoRA template, and action visualization.

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
- A100 80GB is closer to the official LoRA example.
- First cloud experiment should be the official LIBERO-Spatial checkpoint smoke test with `NUM_TRIALS_PER_TASK=1`.
- Decide whether to run LoRA only after the smoke test, disk space, CUDA PyTorch setup, and dataset registration are confirmed.

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
NUM_TRIALS_PER_TASK=1 bash scripts/run_libero_eval.sh
```

Then save the log path in:

```text
notes/experiment_log.md
```

Commit the resulting logs/notes only after reviewing what was actually run.

## LIBERO Smoke Test

The smoke-test evaluation script defaults to:

```bash
CHECKPOINT=openvla/openvla-7b-finetuned-libero-spatial
TASK_SUITE=libero_spatial
NUM_TRIALS_PER_TASK=1
CENTER_CROP=True
bash scripts/run_libero_eval.sh
```

The first real run may download the selected Hugging Face checkpoint. Do this only on a suitable Linux cloud GPU.

## LoRA Template

The LoRA script is a conservative template and does not start training unless explicitly confirmed:

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

Do not start with full LIBERO evaluation.

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
- [ ] Record all experiments in `notes/experiment_log.md`.
- [ ] Prepare resume-ready bullets in `notes/resume_bullets.md`.
