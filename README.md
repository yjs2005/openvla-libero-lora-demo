# OpenVLA LIBERO LoRA Demo

This repository is a robotics research portfolio project for reproducing OpenVLA inference evaluation on LIBERO manipulation tasks, then extending it with small-scale LoRA fine-tuning, action-sequence visualization, and experiment tracking.

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

## Environment Check

Run:

```bash
bash scripts/check_env.sh
```

The report will be written to:

```text
notes/environment_report.md
```

## Safety Notes

Current repository scripts do not automatically download large models.

Current repository scripts do not automatically download large datasets.

Current repository scripts do not automatically run training.

Large downloads, LIBERO setup, OpenVLA checkpoints, and training commands should only be added after explicit confirmation.

## TODO

- [x] Create repository skeleton.
- [x] Add repeatable environment check.
- [ ] Confirm target Python version and package manager.
- [ ] Add official OpenVLA setup commands.
- [ ] Add LIBERO installation and dataset instructions.
- [ ] Add small debug evaluation command.
- [ ] Add LoRA fine-tuning command with conservative defaults.
- [ ] Add action visualization utilities.
- [ ] Record all experiments in `notes/experiment_log.md`.
- [ ] Prepare resume-ready bullets in `notes/resume_bullets.md`.
