# Experiment Log

Record only experiments that were actually run. Do not write results for planned or failed-to-start commands.

Official checkpoint results and self-trained LoRA results must be recorded separately.

Smoke tests and full evaluations must be recorded separately.

## Table

| Date | Phase | Machine / GPU | Git commit | Model checkpoint | Dataset / task suite | Command | Key hyperparameters | Result | Log path | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-27 | Phase 3 smoke test | H800 PCIe 80GB | `a8e41f5` plus remote hotfixes | `openvla-7b-finetuned-libero-spatial` | `libero_spatial` | `CHECKPOINT=/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial NUM_TRIALS_PER_TASK=1 bash -x scripts/run_libero_eval.sh` | `MAX_TASKS=1`, `MAX_STEPS_PER_EPISODE=5` | Smoke chain passed; `success=False`, `0/1` | Remote historical debug logs | Step-limited smoke test only; not a model performance result. |
| 2026-06-27 | Phase 3 render stability | H800 PCIe 80GB | `a8e41f5` plus remote hotfixes | `openvla-7b-finetuned-libero-spatial` | `libero_spatial` | `RENDER_BACKEND=osmesa SAVE_VIDEO=1 MAX_TASKS=1 MAX_STEPS_PER_EPISODE=full NUM_TRIALS_PER_TASK=1 bash -x scripts/run_libero_eval.sh` | Compared EGL vs OSMesa; video on/off | OSMesa completed 100-step, 200-step, and single-task full rollout with `success=True`; EGL crashed around 51 action steps on 100-step runs | Remote historical render-debug logs | OSMesa chosen as stable evaluation backend. |
| 2026-06-27 | Phase 3 official baseline | H800 PCIe 80GB | `a8e41f5` plus remote hotfixes | `openvla-7b-finetuned-libero-spatial` | `libero_spatial` | `RENDER_BACKEND=osmesa SAVE_VIDEO=1 MAX_TASKS=1 MAX_STEPS_PER_EPISODE=full NUM_TRIALS_PER_TASK=3 bash -x scripts/run_libero_eval.sh` | 1 task, 3 trials, full rollout, OSMesa | `3/3` success; action steps: 81, 107, 77 | `artifacts/openvla_official_baseline_logs_latest.tar.gz` | Official checkpoint baseline; MP4 files saved in `artifacts/rollouts/`. |
| 2026-06-27 | Phase 3 official baseline | H800 PCIe 80GB | `a8e41f5` plus remote hotfixes | `openvla-7b-finetuned-libero-spatial` | `libero_spatial` | `RENDER_BACKEND=osmesa SAVE_VIDEO=1 MAX_TASKS=3 MAX_STEPS_PER_EPISODE=full NUM_TRIALS_PER_TASK=3 bash -x scripts/run_libero_eval.sh` | 3 tasks, 3 trials each, full rollout, OSMesa | `9/9` success; no native abort, EGL, or `read_pixels` crash | `artifacts/openvla_official_baseline_logs_latest.tar.gz` | Reference baseline for later LoRA comparison. |

## Rules

- 未运行的实验不能写结果。
- 官方 checkpoint 结果和自己 LoRA 训练结果要分开记录。
- smoke test 和 full evaluation 要分开记录。
- Result should include only measured outputs from logs, not expectations from papers or README files.

## 2026-06-27 OpenVLA + LIBERO official checkpoint smoke test

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Checkpoint: `openvla-7b-finetuned-libero-spatial`, fully downloaded on the remote server at `/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial`.
- Model loading: successful. The log contains `Loading checkpoint shards: 100%` and `Loaded model`.
- LIBERO task suite: successfully entered `Task suite: libero_spatial`.
- Smoke rollout: executed 1 task, 1 episode, and 5 action steps.
- Rollout artifact: saved remotely and downloaded locally to `artifacts/rollouts/openvla_libero_latest_smoke.mp4`.
- Result: `success=False`, `0/1`. Because this was a step-limited smoke test, this must not be interpreted as model performance.
- The early local smoke-test artifact was cleaned after the official baseline was consolidated.

## 2026-06-27 Render stability debugging

- EGL 50-step rollout: completed and saved MP4, but longer runs were unstable.
- EGL 100-step rollout with video disabled: failed at 51 action steps with `Aborted (core dumped)`.
- OSMesa 100-step rollout with video disabled: completed with `success=True`.
- OSMesa 100-step rollout with video enabled: completed with `success=True` and saved MP4.
- OSMesa 200-step rollout with video enabled: completed with `success=True` and saved MP4.
- OSMesa single-task full rollout with video enabled: completed with `success=True` and saved MP4.
- Conclusion: use `RENDER_BACKEND=osmesa` for later LIBERO evaluation on this cloud instance.

## 2026-06-27 OpenVLA + LIBERO official checkpoint baseline

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Checkpoint: `/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial`.
- Renderer: OSMesa.
- Task suite: `libero_spatial`.
- Video saving: enabled.
- Local log bundle: `artifacts/openvla_official_baseline_logs_latest.tar.gz`.
- Local rollout videos: `artifacts/rollouts/`.

### 1 task x 3 trials

Task: pick up the black bowl between the plate and the ramekin and place it on the plate.

| Episode | Success | Action steps |
| --- | --- | --- |
| 1 | True | 81 |
| 2 | True | 107 |
| 3 | True | 77 |

Success rate: `3/3 = 100%`.

### 3 tasks x 3 trials

| Task | Episodes | Success rate | Action steps |
| --- | --- | --- | --- |
| Pick up the black bowl between the plate and the ramekin and place it on the plate | 3 | `3/3 = 100%` | 81, 107, 77 |
| Pick up the black bowl next to the ramekin and place it on the plate | 3 | `3/3 = 100%` | 113, 95, 140 |
| Pick up the black bowl from table center and place it on the plate | 3 | `3/3 = 100%` | 206, 103, 101 |

Overall success rate: `9/9 = 100%`.

No native abort, EGL, `read_pixels`, segmentation fault, or core dump occurred in the OSMesa baseline runs.

This is the official checkpoint baseline and will be used as the reference point for later LoRA comparison.

## 2026-06-27 LoRA smoke dry-runs

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Base checkpoint: `/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial`.
- Dataset: `libero_spatial_no_noops` RLDS/TFDS data under `/root/autodl-tmp/openvla-libero-lora-demo/data/rlds`.
- Training entry: `external/openvla/vla-scripts/finetune.py`.
- LoRA configuration: rank `32`, dropout `0.0`, batch size `1`, gradient accumulation `1`, learning rate `5e-4`, `image_aug=False`.
- Trainable parameters: `110,828,288 / 7,652,065,472 = 1.4483%`.

### No-save dry-run

- Command intent: `max_steps=2`, `save_steps=1000`, no adapter or merged model output expected.
- Result: completed; data loading, model loading, LoRA insertion, forward/backward pass, and optimizer update were verified.
- Metrics:

| Step | Loss | Action accuracy | Action L1 loss |
| --- | --- | --- | --- |
| 0 | 0.000183 | 1.000000 | 0.000000 |
| 1 | 0.004093 | 1.000000 | 0.000000 |
| 2 | 0.603366 | 0.857143 | 0.001120 |

- Checkpoint output: no `.safetensors` or `.bin` files were written.
- Log path: `results/logs/lora_no_save_metrics.log`.

### Save-enabled dry-run

- Command intent: `max_steps=2`, `save_steps=1`, verify save behavior only.
- Result: completed; adapter and merged HuggingFace model were both saved.
- Metrics:

| Step | Loss | Action accuracy | Action L1 loss |
| --- | --- | --- | --- |
| 0 | 0.014726 | 1.000000 | 0.000000 |
| 1 | 0.007564 | 1.000000 | 0.000000 |
| 2 | 1.159093 | 0.857143 | 0.085154 |

- Adapter output: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke`, size `463M`.
- Merged model output: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke`, size `15G`.
- Log path: `results/logs/lora_save_smoke.log`.

These runs are smoke tests for the fine-tuning pipeline. They do not demonstrate performance improvement over the official checkpoint baseline.

## 2026-06-27 Save-smoke merged model evaluation

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Model checkpoint: merged HuggingFace model from the 2-step save-enabled LoRA smoke run.
- Checkpoint path: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke`.
- Adapter path: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_save_smoke`.
- Lightweight load check: `AutoConfig` and `AutoProcessor` both loaded successfully from the merged model directory.
- Eval setting: `RENDER_BACKEND=osmesa`, `MAX_TASKS=1`, `NUM_TRIALS_PER_TASK=1`, `MAX_STEPS_PER_EPISODE=full`, `SAVE_VIDEO=1`.
- Result: completed `1 task x 1 trial`; `success=True`; action steps: `74`.
- Rollout MP4: `/root/autodl-tmp/openvla-libero-lora-demo/external/openvla/rollouts/2026_06_27/2026_06_27-16_30_21--episode=1--success=True--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4`.
- Log path: `results/logs/eval_lora_save_smoke_1task_1trial.log`.
- Stability: no native abort, `read_pixels`, OOM, CUDA crash, segmentation fault, or core dump occurred.

This evaluation only verifies that the saved 2-step smoke model can be loaded and rolled out. It should not be reported as a LoRA performance improvement.

## 2026-06-27 50-step LoRA small training and evaluation

- Machine / GPU: AutoDL cloud instance, NVIDIA H800 PCIe 80GB.
- Base checkpoint: `/root/autodl-tmp/openvla_checkpoints/openvla-7b-finetuned-libero-spatial`.
- Dataset: `libero_spatial_no_noops`.
- Training entry: `external/openvla/vla-scripts/finetune.py`.
- Training setting: `max_steps=50`, `save_steps=50`, `batch_size=4`, `grad_accumulation_steps=1`, `learning_rate=5e-4`, `lora_rank=32`, `lora_dropout=0.0`, `image_aug=True`.
- Trainable parameters: `110,828,288 / 7,652,065,472 = 1.4483%`.
- Adapter output: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_adapters_tmp/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_lora50--image_aug`, size `463M`.
- Merged model output: `/root/autodl-tmp/openvla-libero-lora-demo/results/lora_runs/openvla-7b-finetuned-libero-spatial+libero_spatial_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--libero_spatial_lora50--image_aug`, size `15G`.
- Training log: `results/logs/lora_spatial_50steps.log`.

Final representative training metrics:

| Step | Loss | Action accuracy | Action L1 loss |
| --- | --- | --- | --- |
| 43 | 0.696080 | 0.928571 | 0.020448 |
| 44 | 1.438865 | 0.892857 | 0.010644 |
| 45 | 1.086499 | 0.857143 | 0.028011 |
| 46 | 2.133811 | 0.821429 | 0.018207 |
| 47 | 0.245641 | 0.964286 | 0.001120 |
| 48 | 0.383422 | 0.857143 | 0.006723 |
| 49 | 0.518494 | 0.892857 | 0.009244 |
| 50 | 1.334518 | 0.892857 | 0.000840 |

### LoRA-50 evaluation

Renderer: OSMesa.

`1 task x 1 trial`:

| Episode | Success | Action steps |
| --- | --- | --- |
| 1 | True | 83 |

`3 tasks x 3 trials`:

| Task | Episodes | Success rate | Action steps |
| --- | --- | --- | --- |
| Pick up the black bowl between the plate and the ramekin and place it on the plate | 3 | `3/3` | 83, 74, 89 |
| Pick up the black bowl next to the ramekin and place it on the plate | 3 | `3/3` | 112, 115, 132 |
| Pick up the black bowl from table center and place it on the plate | 3 | `2/3` | 220, 115, 96 |

Overall success rate: `8/9 = 88.9%`.

Stability: no OOM, CUDA crash, native abort, `read_pixels`, segmentation fault, core dump, or Python traceback occurred.

Comparison:

- Official checkpoint baseline on the same 3-task subset: `9/9`.
- LoRA-50 on the same 3-task subset: `8/9`.
- This run should be described as a small-scale LoRA workflow validation, not as a performance improvement.
- Failure case: episode 7, task "pick up the black bowl from table center and place it on the plate", `success=False`, 220 action steps. Inspect this rollout video before expanding training.
- Next recommendation: pause larger training; if continuing, prefer a lower learning rate before increasing to 200 steps.
