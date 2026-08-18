# Krum CIFAR-10 Reproduction

本项目旨在复现论文 *Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent* 中的 Krum 和 Multi-Krum 拜占庭鲁棒梯度聚合方法，并在
CIFAR-10 上串行模拟由 10 个 worker 参与的分布式 SGD 训练。

**当前状态：** 已完成环境配置、CIFAR-10 单机 baseline，以及使用 Mean 聚合的
10-worker 串行分布式 SGD。

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
Krum、Multi-Krum 和 Byzantine attacks。

## 10-worker 串行分布式 SGD

当前已实现单进程内的 10-worker 同步 distributed SGD 模拟。CIFAR-10 训练集
被确定性地随机划分为 10 个 IID shard；每轮各 worker 基于同一个 global model
串行计算梯度，parameter server 使用 Mean 聚合，并只更新一次 global model。

```bash
conda activate krum
python train_distributed.py --epochs 1
```

可以使用少量 round 快速验证完整数据流：

```bash
python train_distributed.py --epochs 1 --max-rounds 2
```

当前尚未实现 Krum、Multi-Krum 和 Byzantine attacks。

### Mean baseline 的经验训练长度

在当前默认模型和以下训练配置下：

```text
workers = 10
batch_size_per_worker = 50
optimizer = SGD
learning_rate = 0.05
```

每个 worker 拥有 5000 个训练样本，因此每个 epoch 包含 100 个 global rounds。
实际运行结果显示，测试准确率在约 4000 rounds 后开始进入平台期，到约 6000
rounds 时已经基本收敛；继续训练到 10000 rounds 并没有进一步提高测试准确率，
同时测试 loss 明显上升，表现出过拟合。因此，后续 Mean、Krum 及攻击实验建议统一
使用 6000 global rounds，既能让当前 baseline 充分训练，也便于公平比较收敛曲线。

对应 6000 rounds 的运行命令为：

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name mean-clean-6000rounds-seed0
```

一次 10000-round 参考实验中，最佳测试准确率为 72.71%，出现在 round 3900；
round 6000 的测试准确率为 71.95%，round 10000 为 72.05%。虽然 6000 rounds
之后准确率基本稳定在 72% 左右，但测试 loss 从 round 4000 的 0.9110 上升到
round 10000 的 1.9074，因此不建议只通过增加 epoch 继续延长后续对比实验。

## 实验记录与结果对比

`train_distributed.py` 默认把每次实验记录到独立目录：

```text
results/<timestamp>_<run-name>/
├── config.json
├── rounds.csv
├── metrics.csv
├── aggregation.jsonl
└── summary.json
```

其中 `config.json` 保存运行参数和 Git 状态，`rounds.csv` 保存逐 round 的训练
指标，`metrics.csv` 保存定期测试指标，`aggregation.jsonl` 保存聚合器选择的 worker，
`summary.json` 保存最终结果和实验是否完整结束。整个 `results/` 目录已被 Git 忽略。

运行当前无攻击的 Mean baseline，并每 10 个 global round 测试一次：

```bash
conda activate krum
python train_distributed.py \
  --epochs 5 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --eval-interval-rounds 10 \
  --seed 0 \
  --run-name mean-clean-b50-lr005-seed0
```

如果不希望记录文件，可以使用：

```bash
python train_distributed.py --epochs 1 --no-record
```

完成多个实验后，解析 `results/` 下的全部实验并生成对比图：

```bash
python plot_results.py \
  --results-dir results \
  --output results/comparison.png
```

也可以只比较指定的实验目录：

```bash
python plot_results.py \
  results/MEAN_RUN_DIRECTORY \
  results/KRUM_RUN_DIRECTORY \
  --output results/mean-vs-krum.png
```

相同配置使用多个随机种子运行后，可按配置聚合曲线，并绘制均值及一个标准差范围：

```bash
python plot_results.py \
  --results-dir results \
  --aggregate-seeds \
  --output results/comparison-by-condition.png
```

当前训练入口只实现 Mean 且没有 Byzantine worker；记录格式和绘图脚本已经为后续
Krum、Multi-Krum 及攻击实验预留了 `aggregator`、`attack`、
`num_byzantine`、`selected_worker_ids` 和 `selected_byzantine` 等字段。
