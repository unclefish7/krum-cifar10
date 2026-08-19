# Krum CIFAR-10 Reproduction

本项目用于复现论文 *Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent* 中的 Krum / Multi-Krum 梯度聚合方法，并在 CIFAR-10 上通过
单个 Python 进程串行模拟 10-worker 同步分布式 SGD。

## 当前状态

已经完成：

- CIFAR-10 数据加载与 IID worker 数据划分
- 小型 `SimpleCNN` 和单机训练 baseline
- 10-worker 串行同步 distributed SGD
- Mean 聚合
- Krum 聚合及 `--aggregator` 选择接口
- Gaussian Byzantine attack
- 实验配置、逐 round 指标和测试指标记录
- 多次实验结果解析与对比绘图

尚未实现：

- 其他 Byzantine attacks
- Multi-Krum
- non-IID 数据划分

Krum 已在真实 GPU 上完成 6000-round 无攻击训练，最佳测试准确率为 69.39%。
Gaussian attack 已完成实现，但尚未进行完整训练验证。

## 仓库结构

```text
data.py                 CIFAR-10 加载和 10-worker IID 划分
model.py                SimpleCNN
train_baseline.py       单机 Adam baseline
distributed.py          梯度操作、Mean 和 Krum
attacks.py              Gaussian Byzantine gradient attack
train_distributed.py    10-worker 串行同步训练入口
experiment_logger.py    实验结果记录器
plot_results.py         多实验结果解析与绘图
requirements.txt        基础 Python 依赖
docs/                   论文 PDF
```

训练数据通过仓库中的 `./data` 访问。当前机器使用软链接将其指向：

```text
~/datasets/cifar10
```

`data/`、`results/` 和 Python 缓存均不进入 Git 版本控制。

## 环境

- WSL2 / Linux
- Python 3.10.20
- Conda 环境：`krum`
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- PyTorch CUDA 运行时：12.8
- NVIDIA 驱动：610.74
- 驱动 CUDA compatibility：13.3
- GPU：NVIDIA GeForce RTX 4070 Ti SUPER（16376 MiB）

创建并激活环境：

```bash
conda create -n krum python=3.10 pip -y
conda activate krum
```

安装 PyTorch 官方 CUDA 12.8 wheel。wheel 已包含所需 CUDA 运行时，不需要另行
安装完整 CUDA Toolkit：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装其余基础依赖：

```bash
python -m pip install -r requirements.txt
```

检查 PyTorch、CUDA 和 GPU：

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 运行训练

开始前先激活环境：

```bash
conda activate krum
```

### 单机 baseline

```bash
python train_baseline.py --epochs 5 --batch-size 128
```

该入口使用完整 CIFAR-10 训练集、`SimpleCNN` 和 Adam，主要用于验证数据、模型和
GPU 训练闭环。正式比较 Mean 与 Krum 时，应使用下面的 10-worker 训练入口，而
不是将单机 Adam 结果直接与 Krum 比较。

### 快速检查 10-worker 数据流

下面的命令默认使用 Mean，只执行两个 global rounds：

```bash
python train_distributed.py \
  --epochs 1 \
  --max-rounds 2 \
  --batch-size 50 \
  --run-name mean-quick-check
```

### Mean 无攻击实验

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator mean \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name mean-clean-6000rounds-seed0
```

### Krum 无攻击实验

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator krum \
  --krum-f 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name krum-clean-6000rounds-seed0
```

`--krum-f` 表示 Krum 被配置为最多容忍多少个 Byzantine worker，它不会创建攻击
或恶意 worker。Krum 的参数必须满足：

```text
2 * krum_f + 2 < num_workers
```

当前 `num_workers = 10`，因此 `krum_f` 最大为 3。无攻击实验使用
`--krum-f 0`。

### Gaussian attack 实验

Gaussian Byzantine worker 不提交正常梯度，而是提交均值为 0、标准差为 200 的
随机向量。`--byzantine-selection` 支持两种恶意 worker 选择方式：

```text
fixed    整次实验固定使用 worker 0 到 worker f-1，符合论文的固定 Byzantine 进程模型
round    每个 global round 重新随机选择 f 个不同 worker，作为额外的动态攻击实验
```

随机选择使用独立的 `--byzantine-selection-seed`，因此 Mean 和 Krum 只要使用相同
seed，就会得到相同的逐 round 恶意 worker 序列。`--attack-seed` 则单独控制替换后
的高斯向量。下面的命令使用每 round 随机选择；若要复现论文的固定进程模型，只需
将 `--byzantine-selection round` 改为 `--byzantine-selection fixed`。

Mean + Gaussian attack：

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator mean \
  --attack gaussian \
  --num-byzantine 3 \
  --byzantine-selection round \
  --byzantine-selection-seed 0 \
  --attack-std 200 \
  --attack-seed 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name mean-gaussian-f3-round-6000rounds-seed0
```

Krum + Gaussian attack：

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator krum \
  --krum-f 3 \
  --attack gaussian \
  --num-byzantine 3 \
  --byzantine-selection round \
  --byzantine-selection-seed 0 \
  --attack-std 200 \
  --attack-seed 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name krum-gaussian-f3-round-6000rounds-seed0
```

攻击发生在 worker 梯度 stack 完成后、聚合器运行前。实现会保留诚实梯度，复制
梯度矩阵并替换恶意 worker 对应的行，不会原地破坏攻击前数据。攻击实验中的训练
loss 和 accuracy 只统计当前 round 的诚实 worker。`aggregation.jsonl` 会逐 round
保存实际恶意 worker ID，并记录聚合器是否选择了恶意 worker。

## 推荐实验预算

当前推荐配置为：

```text
num_workers = 10
batch_size_per_worker = 50
optimizer = SGD
learning_rate = 0.05
global_rounds = 6000
```

每个 worker 拥有 5000 个训练样本，batch size 为 50，因此：

```text
每个 epoch = 100 global rounds
60 epochs = 6000 global rounds
```

Mean 无攻击的 10000-round 参考实验结果如下：

| Global round | Test accuracy | Test loss |
| ---: | ---: | ---: |
| 4000 | 72.11% | 0.9110 |
| 6000 | 71.95% | 1.2008 |
| 10000 | 72.05% | 1.9074 |

该次实验的最佳测试准确率为 72.71%，出现在 round 3900。测试准确率约在 4000
rounds 后进入平台，到 6000 rounds 已基本稳定；继续训练到 10000 rounds 没有提高
准确率，同时测试 loss 明显上升。因此后续 Mean、Krum 和攻击实验统一使用 6000
global rounds，主要通过完整曲线比较收敛速度和稳定性，而不是只比较最终点。

## 实验记录

`train_distributed.py` 默认在 `results/` 下为每次运行创建独立目录：

```text
results/<timestamp>_<run-name>/
├── config.json
├── rounds.csv
├── metrics.csv
├── aggregation.jsonl
└── summary.json
```

- `config.json`：训练配置、聚合器、攻击参数、随机种子、环境和 Git 状态
- `rounds.csv`：逐 round 的训练 loss、accuracy、梯度范数和耗时
- `metrics.csv`：定期测试的 loss、accuracy 和 error
- `aggregation.jsonl`：聚合器选择的 worker；Krum 还会记录被选中 worker 的 score
- `summary.json`：运行是否完成、最终指标、总 rounds 和总耗时

如果不需要保存记录：

```bash
python train_distributed.py --epochs 1 --no-record
```

整个 `results/` 目录已加入 `.gitignore`，实验记录和生成的图表不会被提交到 Git。

## 结果对比与绘图

解析 `results/` 下的全部实验并生成对比图：

```bash
python plot_results.py \
  --results-dir results \
  --output results/comparison.png
```

只比较指定的 Mean 和 Krum 实验：

```bash
python plot_results.py \
  results/MEAN_RUN_DIRECTORY \
  results/KRUM_RUN_DIRECTORY \
  --output results/mean-vs-krum.png
```

使用多个随机种子后，可以按相同配置聚合曲线，并绘制均值及一个标准差范围：

```bash
python plot_results.py \
  --results-dir results \
  --aggregate-seeds \
  --output results/comparison-by-condition.png
```

对比图包含：

- test error 与 test loss
- 每个 round 的训练 loss
- 聚合梯度范数
- 聚合耗时
- 被选中 Byzantine worker 的比例

## 后续计划

1. 运行并验证 Mean / Krum 的 Gaussian attack 实验
2. 对比 Mean / Krum 在无攻击与有攻击情况下的曲线
3. 实现 Multi-Krum
4. 评估其他 Byzantine attacks
5. 扩展实验对比与结果整理
