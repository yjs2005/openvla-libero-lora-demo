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
