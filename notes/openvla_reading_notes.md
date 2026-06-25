# OpenVLA Reading Notes

Source checked: official OpenVLA repository, `https://github.com/openvla/openvla`, main branch README and source files.

## Current Stage

- This project is currently in code-reading and script-preparation mode.
- OpenVLA model checkpoints have not been downloaded.
- LIBERO datasets have not been downloaded.
- LIBERO evaluation has not been run.
- LoRA fine-tuning has not been run.
- No experimental result should be claimed until logs exist under `results/logs/` and an entry is added to `notes/experiment_log.md`.

## Project Goal

OpenVLA is a Vision-Language-Action model. It takes robot observation images and a language instruction as input, then predicts robot actions for manipulation tasks. In this project, the goal is to reproduce the official OpenVLA LIBERO evaluation workflow first, then prepare a conservative LoRA fine-tuning workflow and action-sequence visualization for a robotics research portfolio.

## Official Repository Paths

| Function | Official path | Notes |
| --- | --- | --- |
| LoRA fine-tuning entry | `vla-scripts/finetune.py` | Official LoRA training entry point. Uses Hugging Face AutoClasses plus PEFT `LoraConfig`. |
| LIBERO evaluation entry | `experiments/robot/libero/run_libero_eval.py` | Runs OpenVLA in LIBERO simulation environments. Default official evaluation is 50 trials per task; this repo uses smoke-test templates first. |
| LIBERO utilities | `experiments/robot/libero/libero_utils.py` | Creates LIBERO environments, extracts images, converts quaternions, and saves rollout videos. |
| Model loading for robot eval | `experiments/robot/robot_utils.py` | Provides `get_model()` and `get_action()` wrappers used by robot evaluation scripts. |
| OpenVLA HF loading/action wrapper | `experiments/robot/openvla_utils.py` | Loads checkpoints with `AutoModelForVision2Seq.from_pretrained()` and calls `vla.predict_action(...)`. |
| Dataset loading | `prismatic/vla/datasets/rlds/dataset.py` | Core RLDS dataset initializer; loads TFDS/RLDS datasets, computes statistics, applies trajectory/frame transforms. |
| RLDS dataset config | `prismatic/vla/datasets/rlds/oxe/configs.py` | Registers dataset names, observation keys, state/action encodings. Includes `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, and `libero_10_no_noops`. |
| RLDS dataset transforms | `prismatic/vla/datasets/rlds/oxe/transforms.py` | Holds per-dataset standardization transforms. New RLDS datasets must register a config and transform. |
| RLDS mixtures | `prismatic/vla/datasets/rlds/oxe/mixtures.py` | Defines Open X-Embodiment dataset mixtures and sampling weights. |
| Action tokenization | `prismatic/vla/action_tokenizer.py` | Discretizes continuous robot actions into language-model tokens and decodes token IDs back to normalized continuous actions. |
| Action prediction | `prismatic/extern/hf/modeling_prismatic.py` | `OpenVLAForActionPrediction.predict_action()` runs generation, decodes action tokens, and unnormalizes actions. |
| Action normalization | `prismatic/vla/datasets/rlds/utils/data_utils.py` | `normalize_action_and_proprio()` supports `normal`, `bounds`, and `bounds_q99` normalization. |
| Troubleshooting | `README.md`, sections `VLA Troubleshooting` and `VLA Performance Troubleshooting` | Notes package version issues, TFDS/dlimp fixes, and sanity checks for fine-tuning data and inference pipelines. |

## Key Entry Points

- LoRA fine-tuning script: `vla-scripts/finetune.py`
- LIBERO evaluation script: `experiments/robot/libero/run_libero_eval.py`
- Official LIBERO-Spatial evaluation command uses:
  - `--model_family openvla`
  - `--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial`
  - `--task_suite_name libero_spatial`
  - `--center_crop True`

## Official LoRA Parameters Noted

The official LoRA example uses:

- base model: `openvla/openvla-7b`
- `--lora_rank 32`
- `--batch_size 16` in the A100 80GB example
- `--grad_accumulation_steps 1` in the A100 80GB example
- `--learning_rate 5e-4`
- `--image_aug <True or False>`
- `--save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>`
- PEFT config in code:
  - `r=cfg.lora_rank`
  - `lora_alpha=min(cfg.lora_rank, 16)`
  - `lora_dropout=0.0` by default
  - `target_modules="all-linear"`
  - `init_lora_weights="gaussian"`

Official notes indicate that batch size 16 with gradient accumulation 1 needs about 72GB GPU memory, and a smaller GPU should reduce batch size and increase gradient accumulation. The header comment in `vla-scripts/finetune.py` says one 48GB GPU can fit batch size 12, and one 80GB GPU can fit batch size 24.

## Official LIBERO Fine-Tuned Checkpoints

- `openvla/openvla-7b-finetuned-libero-spatial`
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

The official README says each evaluation command automatically downloads the selected Hugging Face checkpoint on first run.

## Hugging Face Token Notes

- Public checkpoint loading may work without a token depending on the Hugging Face model page and environment.
- The official full fine-tuning section documents creating a Hugging Face user access token and placing it in `openvla/.hf_token`.
- Hugging Face dataset clone commands such as `git clone git@hf.co:datasets/openvla/modified_libero_rlds` require Hugging Face access setup and can download large data.
- This project should never commit `.hf_token`; it is ignored by `.gitignore`.

## Large Download / High Memory Steps

- `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b...")` downloads and loads a 7B checkpoint.
- Official LIBERO evaluation downloads one of the LIBERO fine-tuned checkpoints on first run.
- Optional modified LIBERO RLDS dataset clone downloads about 10GB.
- BridgeData V2 download is about 124GB and should not be used at the start of this project.
- LoRA fine-tuning loads OpenVLA-7B and can require tens of GB of VRAM. The official example uses an A100 80GB.
- Full fine-tuning is far more expensive and should not be attempted for this portfolio demo.

## Local vs Cloud Decision

- Local machine from Phase 1: RTX 4060 Laptop GPU, about 8GB VRAM, CPU-only PyTorch detected. This is not recommended for OpenVLA-7B inference or LoRA.
- Recommended cloud start: A100 40GB or better, with A100 80GB closer to the official LoRA example.
- First cloud run should be an official LIBERO-Spatial checkpoint smoke test with `NUM_TRIALS_PER_TASK=1`, not a full 500-rollout evaluation.
