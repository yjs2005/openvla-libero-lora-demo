# Resume Bullets

Use this file only for resume-ready claims supported by code, logs, figures, or notes in this repository. Do not state evaluation or training as completed until corresponding logs exist.

## 当前已完成

- 搭建 OpenVLA-LIBERO-LoRA 机器人 VLA 复现实验仓库，完成项目结构、环境检测脚本、实验记录模板与可视化工具初版。
- 对本地 Windows + RTX 4060 Laptop 8GB 环境进行自动化检测，明确其更适合代码开发、文档整理、轻量可视化与 GitHub 管理，不适合直接运行 OpenVLA-7B。
- 阅读 OpenVLA 官方仓库，整理 LoRA 微调入口、LIBERO 评估入口、RLDS 数据加载、动作归一化与 checkpoint 加载相关代码路径。
- 准备 Linux 云端 GPU 环境初始化脚本、官方 LIBERO-Spatial checkpoint smoke test 脚本和 LoRA 小规模微调模板。

## 云端评估完成后可写

- 基于 OpenVLA 官方 LIBERO-Spatial fine-tuned checkpoint，在云端 GPU 上完成 LIBERO smoke test/评估，并记录运行日志、任务配置和成功率结果。
- 构建 OpenVLA 动作序列可视化流程，将模型输出动作拆分为位移、旋转和夹爪曲线，用于分析策略行为。

## LoRA 微调完成后可写

- 基于 OpenVLA-7B 和 LoRA 方法完成小规模机器人操作任务微调实验，记录训练配置、日志和 checkpoint 管理流程。
- 使用 LIBERO/RLDS 格式数据完成数据注册、加载和小规模训练验证，分析不同任务 suite 下的策略成功率。
- 对比官方 checkpoint 与自训练 LoRA checkpoint 的 LIBERO 表现，结合训练日志、动作误差和成功率进行实验复盘。
