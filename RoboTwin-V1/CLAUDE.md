# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

RoboTwin是一个双臂机器人基准测试项目，采用生成式数字孪生技术。该项目基于SAPIEN物理仿真器，支持多种机器人学习策略的训练和评估，包括Diffusion Policy (DP)、3D Diffusion Policy (DP3)、RDT和OpenPi等。

## 核心架构

- **仿真环境**: 基于SAPIEN 3.0.0b1，支持Vulkan渲染和Ray tracing
- **任务系统**: 17个标准化双臂操作任务，包括抓取、放置、堆叠等
- **相机系统**: 支持多视角RGB-D相机（D435, L515），用于真实机器人对齐
- **数据采集**: 基于专家策略的自动数据收集系统
- **策略框架**: 支持多种深度学习策略的统一接口

## 主要目录结构

```
envs/           - 任务环境定义，每个任务继承base_task.py
script/         - 核心脚本：数据处理、策略评估、任务运行
policy/         - 各种策略实现（DP, DP3, RDT, OpenPi）
task_config/    - 任务配置文件（YAML格式）
data/           - 数据存储目录
models/         - 3D模型资源
```

## 常用命令

### 环境安装
```bash
# 创建conda环境
conda create -n RoboTwin python=3.8
conda activate RoboTwin

# 安装依赖
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0

# 安装pytorch3d
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..

# 下载资源
python ./script/download_asset.py
unzip aloha_urdf.zip && unzip main_models.zip
```

### 修改mplib库代码（必需步骤）
需要手动修改已安装的mplib库代码：
- 移除`mplib/planner.py`第71行的`convex=True`参数
- 移除`mplib/planner.py`第848行的`or collide`条件

### 任务运行和数据采集
```bash
# 运行任务并采集数据
bash run_task.sh ${task_name} ${gpu_id}
# 示例: bash run_task.sh block_hammer_beat 0
```

### Diffusion Policy训练和评估
```bash
# 数据预处理
python script/pkl2zarr_dp.py ${task_name} ${head_camera_type} ${expert_data_num}

# 训练
cd policy/Diffusion-Policy
bash train.sh ${task_name} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}

# 评估
bash eval.sh ${task_name} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id}
```

### 3D Diffusion Policy训练和评估
```bash
# 数据预处理
python script/pkl2zarr_dp3.py ${task_name} ${head_camera_type} ${expert_data_num}

# 训练
cd policy/3D-Diffusion-Policy
bash train.sh ${task_name} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}
# 彩色版本使用: bash train_rgb.sh

# 评估
bash eval.sh ${task_name} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id}
# 彩色版本使用: bash eval_rgb.sh
```

## 任务名称映射

| 显示名称 | task_name |
|---------|-----------|
| Block Hammer Beat | block_hammer_beat |
| Bottle Adjust | bottle_adjust |
| Dual Bottles Pick (Easy/Hard) | dual_bottles_pick_easy/hard |
| Dual Shoes Place | dual_shoes_place |
| Mug Hanging (Easy/Hard) | mug_hanging_easy/hard |
| Pick Apple Messy | pick_apple_messy |
| Put Apple Cabinet | put_apple_cabinet |
| Shoe Place | shoe_place |

## 配置说明

### 相机配置
- `head_camera_type`: 头部相机类型，默认D435
- D435: 320×240分辨率，FOV 37°
- L515: 320×180分辨率，FOV 45°

### 重要配置项
- `episode_num`: 采集数据集数量，默认100
- `pcd_down_sample_num`: 点云下采样数量，默认1024
- `dual_arm`: 是否使用双臂，默认true
- `render_freq`: 可视化频率，离屏设备建议设为0

## 策略部署指南

1. 修改`envs/base_task.py`中的TODO部分，确保`policy.get_action(obs)`返回动作序列
2. 修改`script/eval_policy.py`中的TODO部分来初始化你的策略模型
3. 策略接口应符合BaseImagePolicy规范

## 开发注意事项

- 使用`Ctrl + \`终止Python进程（如果`Ctrl + C`无效）
- Vulkan在某些离屏设备上可能不稳定，尝试重新连接`ssh -X`
- 确保GPU支持CUDA 12.1和适当的NVIDIA驱动程序
- 数据采集和训练需要较大存储空间
- 策略评估默认运行100次测试

## 实施细节

- 你只需要管DP这一个策略, DP3,RDT,OpenPi 都不需要做任何改动,忽略他们!