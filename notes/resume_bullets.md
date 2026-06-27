# Resume Bullets

本文件只记录已有代码、日志、视频或实验记录支撑的简历表述。不要把 smoke test 或小步数实验写成完整大规模训练。

## 当前已完成，可如实表述

- 搭建 OpenVLA-LIBERO-LoRA 机器人 VLA 复现实验仓库，完成环境检测脚本、云端运行脚本、实验记录模板、动作可视化工具和 GitHub 项目管理。
- 在本地 Windows + RTX 4060 Laptop 8GB 环境完成自动化检测，明确本地主要用于代码开发、文档整理、轻量可视化和 GitHub 管理，不直接运行 OpenVLA-7B 推理或微调。
- 在 H800 80GB 云端环境搭建 OpenVLA + LIBERO 推理评估链路，成功加载官方 `openvla-7b-finetuned-libero-spatial` checkpoint。
- 排查 LIBERO rollout 中的 EGL/native abort 稳定性问题，验证 OSMesa 渲染路径可稳定完成长 rollout 并保存 MP4。
- 基于官方 OpenVLA LIBERO-Spatial fine-tuned checkpoint 完成小规模 baseline：OSMesa 下 `1 task x 3 trials = 3/3`，`3 tasks x 3 trials = 9/9`，并保存日志和 rollout 视频。
- 准备 LIBERO-Spatial RLDS 数据 `libero_spatial_no_noops`，确认 TFDS 目录结构、训练样本数和 OpenVLA `finetune.py` 数据加载入口。
- 完成 OpenVLA + LIBERO 小规模 LoRA 微调流程验证：基于 LIBERO-Spatial RLDS 数据完成 LoRA adapter 插入、训练指标记录、checkpoint 保存与 OSMesa rollout 评估闭环，并与 official checkpoint baseline 进行小规模对比。
- 完成 50-step LoRA-on-official-checkpoint 小训练：保存 adapter 和 merged HuggingFace model，OSMesa 下 `1 task x 1 trial = 1/1`，`3 tasks x 3 trials = 8/9`；结果用于验证训练评估闭环，不表述为性能提升。

## 可用于保研简历的谨慎版本

- 搭建并维护 OpenVLA + LIBERO 机器人 VLA 复现实验项目，完成云端 GPU 环境检测、官方 checkpoint 推理评估、渲染稳定性排查、LoRA 小规模训练模板与实验日志管理。
- 基于 H800 80GB 云端环境复现 OpenVLA 官方 LIBERO-Spatial checkpoint 的小规模 OSMesa 评估，完成 `3 tasks x 3 trials` baseline，并记录 success rate、action steps 和 rollout MP4。
- 基于 LIBERO-Spatial RLDS 数据完成 OpenVLA LoRA 小规模流程验证，包括 adapter 插入、前向/反向传播、训练指标输出、checkpoint 保存、merged model 加载和 OSMesa rollout 评估。
- 对比 official checkpoint baseline 与 50-step LoRA-on-official-checkpoint 小实验结果，分析小步数 LoRA 对已有策略的扰动，并形成可复现的日志、视频和实验记录。

## 当前不能写的表述

- 不要写 LoRA 提升了模型性能；当前小规模对比是 official baseline `9/9`，LoRA-50 `8/9`。
- 不要写完整复现了 OpenVLA 官方训练结果；当前是 LoRA-on-official-checkpoint 的小规模流程验证。
- 不要写完成真机部署；当前实验范围是 LIBERO 仿真评估。
- 不要写完成大规模训练；当前最长训练记录是 `max_steps=50`。

## 后续扩展后才可表述

- 扩大 LIBERO-Spatial 评估覆盖更多任务和随机种子，统计更稳定的 success rate、action steps 和失败案例。
- 进行更长步数的 LoRA 微调实验，例如 200 steps 或更多，并在相同任务子集上与 official checkpoint baseline 做严格对比。
- 建立 OpenVLA 动作序列可视化分析流程，将模型输出动作拆分为位移、旋转和夹爪曲线，用于分析策略行为。
