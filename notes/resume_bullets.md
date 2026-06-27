# Resume Bullets

Use this file only for resume-ready claims supported by code, logs, figures, or notes in this repository. Do not state evaluation or training as completed until corresponding logs exist.

## 当前已完成，可如实表述

- 搭建 OpenVLA-LIBERO-LoRA 机器人 VLA 复现实验仓库，完成项目结构、环境检测脚本、云端运行脚本、实验记录模板与动作可视化工具。
- 在本地 Windows + RTX 4060 Laptop 8GB 环境完成自动化检测，明确本地机器主要用于代码开发、文档整理、轻量可视化与 GitHub 管理，不直接运行 OpenVLA-7B。
- 在 H800 80GB 云端环境搭建 OpenVLA + LIBERO 推理评估链路，成功加载官方 `openvla-7b-finetuned-libero-spatial` checkpoint。
- 排查 LIBERO rollout 中的 EGL/native abort 稳定性问题，验证 OSMesa 渲染路径可稳定完成长 rollout 并保存 MP4。
- 基于官方 OpenVLA LIBERO-Spatial fine-tuned checkpoint 完成小规模 baseline：`1 task x 3 trials` 达到 `3/3` success，`3 tasks x 3 trials` 达到 `9/9` success，并保存日志与 rollout 视频。

## 云端评估继续扩展后可写

- 扩展 LIBERO-Spatial official checkpoint 评估覆盖更多 tasks/trials，记录 success rate、action step 数、rollout 视频与失败案例。
- 建立 OpenVLA 动作序列可视化流程，将模型输出动作拆分为位移、旋转和夹爪曲线，用于分析策略行为。

## LoRA 微调完成后才可写

- 基于 OpenVLA-7B 和 LoRA 方法完成小规模机器人操作任务微调实验，记录训练配置、训练日志、adapter 保存与评估流程。
- 使用 LIBERO/RLDS 格式数据完成数据注册、加载和 tiny dry-run 训练验证，分析微调前后策略成功率变化。
- 对比官方 checkpoint 与自训练 LoRA adapter 在 LIBERO 任务上的表现，结合训练日志、动作行为和 success rate 进行实验复盘。
