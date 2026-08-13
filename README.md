# Krum CIFAR-10 Reproduction

本项目旨在复现论文 *Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent* 中的 Krum 和 Multi-Krum 拜占庭鲁棒梯度聚合方法，并在
CIFAR-10 上串行模拟由 10 个 worker 参与的分布式 SGD 训练。

**当前状态：** 已完成环境配置和最基础的 CIFAR-10 单机 baseline。

## 环境信息

- WSL2 / Linux
- Python 3.10.20
- Conda 环境：`krum`
- PyTorch 2.11.0+cu128
- PyTorch CUDA 运行时：12.8
- NVIDIA 驱动：610.74
- `nvidia-smi` 报告的驱动 CUDA compatibility：13.3
- GPU：NVIDIA GeForce RTX 4070 Ti SUPER（16376 MiB）

## 环境配置

创建并激活 Conda 环境：

```bash
conda create -n krum python=3.10 pip -y
conda activate krum
```

从 PyTorch 官方 wheel 索引安装支持 CUDA 12.8 的 PyTorch 和 torchvision。
该 wheel 已包含所需的 CUDA 运行时，无需另行安装完整的系统 CUDA Toolkit：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装其余项目依赖：

```bash
python -m pip install -r requirements.txt
```

## GPU 验证

运行以下命令检查 PyTorch 版本、CUDA 运行时、CUDA 可用状态和 GPU 名称：

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Baseline

当前已实现一个使用小型 CNN 的 CIFAR-10 单机训练 baseline。运行方式：

```bash
conda activate krum
python train_baseline.py
```

可以通过命令行调整基本训练参数，例如：

```bash
python train_baseline.py --epochs 5 --batch-size 128
```

首次运行时，torchvision 会自动将 CIFAR-10 下载到 `./data`。当前阶段尚未实现
distributed SGD、Krum、Multi-Krum 和 Byzantine attacks。
