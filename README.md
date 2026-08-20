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
- Multi-Krum 聚合，可配置选择梯度数量 `m`
- Gaussian Byzantine attack
- Omniscient opposite-gradient Byzantine attack
- 固定或逐 round 随机选择恶意 worker
- 实验配置、逐 round 指标和测试指标记录
- 多次实验结果解析与对比绘图

尚未实现：

- 其他 Byzantine attacks
- non-IID 数据划分

Mean、Krum 和 Multi-Krum 均已在真实 GPU 上完成训练验证。已有实验显示，Gaussian
attack 会破坏 Mean，而 Krum 和 Multi-Krum 能过滤明显的高斯恶意梯度并继续收敛。

## 仓库结构

```text
data.py                 CIFAR-10 加载和 10-worker IID 划分
model.py                SimpleCNN
train_baseline.py       单机 Adam baseline
distributed.py          梯度操作、Mean、Krum 和 Multi-Krum
attacks.py              Gaussian 和 Omniscient Byzantine gradient attacks
train_distributed.py    10-worker 串行同步训练入口
experiment_logger.py    实验结果记录器
plot_results.py         多实验结果解析与绘图
run_experiments.sh      多 seed、固定并发数量的批量实验入口
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

## 训练设计与统一配置

开始前先激活环境：

```bash
conda activate krum
```

正式六组对比实验统一使用：

```text
num_workers = 10
batch_size_per_worker = 50
optimizer = SGD
learning_rate = 0.05
epochs = 60
global_rounds = 6000
evaluation_interval = 100 rounds
```

每个 worker 拥有 5000 个训练样本，batch size 为 50，因此每个 epoch 包含 100 个
global rounds，60 epochs 对应 6000 global rounds。

Mean 对全部梯度求平均。Krum 选择 score 最小的一个梯度；Multi-Krum 选择 score
最小的 `m` 个梯度并求平均。Krum 和 Multi-Krum 必须满足：

```text
2 * krum_f + 2 < num_workers
```

当前 `num_workers=10`，所以标准鲁棒配置使用：

```text
krum_f = 3
multi_krum_m = num_workers - krum_f = 7
```

为了严格比较有无攻击，Krum 和 Multi-Krum 的无攻击实验也保持 `krum_f=3`，
Multi-Krum 同时保持 `m=7`。这样配对实验只改变是否注入攻击，不改变聚合规则。

Gaussian attack 使用均值为 0、标准差为 200 的随机梯度。标准攻击实验统一指定：

```text
--num-byzantine 3
--byzantine-selection fixed
--attack-std 200
```

`fixed` 会让 worker 0、1、2 在整次实验中始终为恶意 worker，与论文的固定
Byzantine 进程模型一致。代码仍支持 `round` 模式，但它属于额外的动态攻击实验，
不用于下面的标准六组对比。

## 快速检查

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
  --no-record
```

## 标准六组实验命令

下面六组命令使用 seed 0。多随机种子实验时，将 `--seed`、`--partition-seed`、
攻击实验的 `--attack-seed`，以及 run name 中的 seed 一起改为 1、2。

如果需要一次运行多组实验，可直接使用后文的“批量运行”脚本；这里保留完整单次命令，
便于检查配置或只重跑某一组。

### 1. Mean，无攻击

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

### 2. Krum，无攻击

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator krum \
  --krum-f 3 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name krum-clean-f3-6000rounds-seed0
```

### 3. Multi-Krum，无攻击

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator multi-krum \
  --krum-f 3 \
  --multi-krum-m 7 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name multi-krum-clean-f3-m7-6000rounds-seed0
```

如果省略 `--multi-krum-m 7`，脚本会自动使用 `m = num_workers - krum_f = 7`。

### 4. Mean，Gaussian attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator mean \
  --attack gaussian \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --attack-std 200 \
  --attack-seed 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name mean-gaussian-f3-fixed-6000rounds-seed0
```

### 5. Krum，Gaussian attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator krum \
  --krum-f 3 \
  --attack gaussian \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --attack-std 200 \
  --attack-seed 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name krum-gaussian-f3-fixed-6000rounds-seed0
```

### 6. Multi-Krum，Gaussian attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator multi-krum \
  --krum-f 3 \
  --multi-krum-m 7 \
  --attack gaussian \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --attack-std 200 \
  --attack-seed 0 \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0
```

攻击发生在 worker 梯度 stack 完成后、聚合器运行前。实现会保留诚实梯度，复制
梯度矩阵并替换恶意 worker 对应的行，不会原地破坏攻击前数据。攻击实验中的训练
loss 和 accuracy 只统计当前 round 的诚实 worker。`aggregation.jsonl` 会逐 round
保存实际恶意 worker ID，并记录聚合器是否选择了恶意 worker。

## Omniscient 反向梯度攻击

论文中的 omniscient adversary 掌握一个高精度梯度估计，并提交方向相反、幅度很大的
梯度。本项目在每个 round 使用全部诚实 worker 梯度的平均值作为攻击者的估计：

```text
honest_mean = mean(all honest worker gradients)
malicious_gradient = -attack_scale * honest_mean
```

所有恶意 worker 提交同一个 `malicious_gradient`。论文实验通过全数据集计算高精度
估计；这里使用攻击者已知的当轮诚实梯度均值，避免每个 global round 额外遍历完整
CIFAR-10 训练集，同时保留“知道正确方向并反向放大”的攻击机制。

默认使用：

```text
--attack omniscient
--attack-scale 100
--num-byzantine 3
--byzantine-selection fixed
```

`--attack-scale` 必须为正数。该攻击本身是确定性的，不使用 `--attack-seed`；如果使用
额外的 `round` 恶意 worker 选择模式，worker 身份仍由
`--byzantine-selection-seed` 控制。

### Mean + Omniscient attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator mean \
  --attack omniscient \
  --attack-scale 100 \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name mean-omniscient-f3-fixed-scale100-6000rounds-seed0
```

### Krum + Omniscient attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator krum \
  --krum-f 3 \
  --attack omniscient \
  --attack-scale 100 \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name krum-omniscient-f3-fixed-scale100-6000rounds-seed0
```

### Multi-Krum + Omniscient attack

```bash
python train_distributed.py \
  --epochs 60 \
  --batch-size 50 \
  --learning-rate 0.05 \
  --aggregator multi-krum \
  --krum-f 3 \
  --multi-krum-m 7 \
  --attack omniscient \
  --attack-scale 100 \
  --num-byzantine 3 \
  --byzantine-selection fixed \
  --eval-interval-rounds 100 \
  --seed 0 \
  --partition-seed 0 \
  --run-name multi-krum-omniscient-f3-fixed-m7-scale100-6000rounds-seed0
```

## 批量运行

`run_experiments.sh` 默认批量调度 3 种聚合器在 clean、Gaussian 和 Omniscient
三种条件下的 seed 0、1、2，并允许最多六个训练
进程同时共享 GPU。脚本不会跳过已经完成的实验；再次执行会重新运行全部启用项。

运行前直接编辑脚本顶部：

```bash
SEEDS=(0 1 2)
MAX_JOBS=6
PROGRESS_INTERVAL_SECONDS=10
```

实验列表采用 `aggregator|condition` 格式。保留某行表示运行，注释掉表示不运行：

```bash
EXPERIMENTS=(
  "mean|clean"
  "krum|clean"
  "multi-krum|clean"
  "mean|gaussian"
  "krum|gaussian"
  "multi-krum|gaussian"
  "mean|omniscient"
  "krum|omniscient"
  "multi-krum|omniscient"
)
```

默认任务数量为：

```text
3 aggregators * 3 conditions * 3 seeds = 27 jobs
```

例如只运行 Krum 和 Multi-Krum 的 Gaussian attack：

```bash
EXPERIMENTS=(
  # "mean|clean"
  # "krum|clean"
  # "multi-krum|clean"
  # "mean|gaussian"
  "krum|gaussian"
  "multi-krum|gaussian"
  # "mean|omniscient"
  # "krum|omniscient"
  # "multi-krum|omniscient"
)
```

启动批量实验：

```bash
conda activate krum
./run_experiments.sh
```

脚本会动态维持不超过 `MAX_JOBS` 个任务。每个任务的终端输出单独写入：

```text
results/batch_logs/<run-name>.log
```

因此并发任务的 tqdm 和指标不会混在同一个终端中。可以在另一个 shell 查看某个任务：

```bash
tail -f results/batch_logs/RUN_NAME.log
```

主终端每 10 秒检查一次运行状态，并将同一次检查的所有任务放在两条横线之间。每个
任务都会打印基于已完成 global rounds 的字符进度条，例如：

```text
--------------------------------------------------------------------------------
Progress update: 2026-08-20T15:30:00+08:00
[progress] mean-clean-6000rounds-seed0              [########------------] 40% 2400/6000
[progress] krum-gaussian-f3-fixed-6000rounds-seed0  [######--------------] 30% 1800/6000
--------------------------------------------------------------------------------
```

检查间隔可以通过 `PROGRESS_INTERVAL_SECONDS` 调整。启动和数据加载期间尚未生成
`rounds.csv` 时，任务会暂时显示为 0%。

重新运行同名任务时，训练记录仍会因时间戳不同而保存在新的实验目录中，但对应的
`batch_logs/<run-name>.log` 会被本次输出覆盖。

使用 `MAX_JOBS=1` 可获得更干净的耗时数据；较大的 `MAX_JOBS` 适合提高多 seed 准确率
实验的整体吞吐量。并发运行得到的 aggregation time、round time 和 total time 不应
用于严格的聚合器性能对比。

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

- `config.json`：训练配置、聚合器、Multi-Krum 的 `m`、攻击参数、随机种子、环境和 Git 状态
- `rounds.csv`：逐 round 的训练 loss、accuracy、梯度范数和耗时
- `metrics.csv`：定期测试的 loss、accuracy 和 error
- `aggregation.jsonl`：逐 round 保存全部被选中 worker、对应 score、选中数量，以及
  其中 Byzantine worker 的数量和比例；适用于 Mean、Krum 和 Multi-Krum
- `summary.json`：运行是否完成、最终指标、总 rounds 和总耗时

如果不需要保存记录：

```bash
python train_distributed.py --epochs 1 --no-record
```

整个 `results/` 目录已加入 `.gitignore`，实验记录和生成的图表不会被提交到 Git。

## 结果对比与绘图

`plot_results.py` 每次生成一张包含六个关键指标面板的图片：

- test error
- test loss
- round train loss
- 聚合梯度 L2 norm
- 聚合耗时
- 被选中 Byzantine worker 的比例

下面的示例通过 run name 后缀匹配结果目录。每个通配符应只匹配一个目标实验；如果
同名实验运行过多次，可先执行：

```bash
ls -d results/*RUN_NAME
```

然后将下面的通配符替换成具体目录。

### 所有已记录实验一起对比

这条命令会读取 `results/` 下的所有实验，包括旧配置和快速检查，适合总览，但不适合
直接得出严格对比结论：

```bash
python plot_results.py \
  --results-dir results \
  --title "All recorded experiments" \
  --output results/all-recorded-experiments.png
```

### 标准六组实验一起对比

```bash
python plot_results.py \
  results/*mean-clean-6000rounds-seed0 \
  results/*krum-clean-f3-6000rounds-seed0 \
  results/*multi-krum-clean-f3-m7-6000rounds-seed0 \
  results/*mean-gaussian-f3-fixed-6000rounds-seed0 \
  results/*krum-gaussian-f3-fixed-6000rounds-seed0 \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0 \
  --title "All aggregators: clean vs Gaussian attack" \
  --output results/all-six-experiments-seed0.png
```

### 无攻击时的 Mean、Krum、Multi-Krum 对比

用于比较三种聚合器的收敛速度、最终 error 和聚合耗时：

```bash
python plot_results.py \
  results/*mean-clean-6000rounds-seed0 \
  results/*krum-clean-f3-6000rounds-seed0 \
  results/*multi-krum-clean-f3-m7-6000rounds-seed0 \
  --title "Mean vs Krum vs Multi-Krum without attack" \
  --output results/clean-aggregators-seed0.png
```

### Gaussian attack 下的三种聚合器对比

用于观察 Mean 是否失效，以及 Krum 和 Multi-Krum 是否继续收敛：

```bash
python plot_results.py \
  results/*mean-gaussian-f3-fixed-6000rounds-seed0 \
  results/*krum-gaussian-f3-fixed-6000rounds-seed0 \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0 \
  --title "Mean vs Krum vs Multi-Krum under Gaussian attack" \
  --output results/gaussian-aggregators-seed0.png
```

### Mean 的有无攻击对比

```bash
python plot_results.py \
  results/*mean-clean-6000rounds-seed0 \
  results/*mean-gaussian-f3-fixed-6000rounds-seed0 \
  --title "Mean: clean vs Gaussian attack" \
  --output results/mean-clean-vs-gaussian-seed0.png
```

### Krum 的有无攻击对比

两个实验都使用 `krum_f=3`，所以差别只来自攻击：

```bash
python plot_results.py \
  results/*krum-clean-f3-6000rounds-seed0 \
  results/*krum-gaussian-f3-fixed-6000rounds-seed0 \
  --title "Krum: clean vs Gaussian attack" \
  --output results/krum-clean-vs-gaussian-seed0.png
```

### Multi-Krum 的有无攻击对比

两个实验都使用 `krum_f=3, m=7`：

```bash
python plot_results.py \
  results/*multi-krum-clean-f3-m7-6000rounds-seed0 \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0 \
  --title "Multi-Krum: clean vs Gaussian attack" \
  --output results/multi-krum-clean-vs-gaussian-seed0.png
```

### Krum 与 Multi-Krum 的完整对比

同时比较两种鲁棒聚合器在有无攻击时的收敛、聚合耗时和恶意梯度选择比例：

```bash
python plot_results.py \
  results/*krum-clean-f3-6000rounds-seed0 \
  results/*multi-krum-clean-f3-m7-6000rounds-seed0 \
  results/*krum-gaussian-f3-fixed-6000rounds-seed0 \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0 \
  --title "Krum vs Multi-Krum: clean and Gaussian attack" \
  --output results/krum-vs-multi-krum-seed0.png
```

只比较两者在 Gaussian attack 下的表现：

```bash
python plot_results.py \
  results/*krum-gaussian-f3-fixed-6000rounds-seed0 \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed0 \
  --title "Krum vs Multi-Krum under Gaussian attack" \
  --output results/krum-vs-multi-krum-gaussian-seed0.png
```

### Omniscient attack 对比

比较 Mean、Krum 和 Multi-Krum 在反向梯度攻击下的表现：

```bash
python plot_results.py \
  results/*mean-omniscient-f3-fixed-scale100-6000rounds-seed0 \
  results/*krum-omniscient-f3-fixed-scale100-6000rounds-seed0 \
  results/*multi-krum-omniscient-f3-fixed-m7-scale100-6000rounds-seed0 \
  --title "Mean vs Krum vs Multi-Krum under omniscient attack" \
  --output results/omniscient-aggregators-seed0.png
```

比较 Krum 和 Multi-Krum 在无攻击及反向梯度攻击下的表现：

```bash
python plot_results.py \
  results/*krum-clean-f3-6000rounds-seed0 \
  results/*multi-krum-clean-f3-m7-6000rounds-seed0 \
  results/*krum-omniscient-f3-fixed-scale100-6000rounds-seed0 \
  results/*multi-krum-omniscient-f3-fixed-m7-scale100-6000rounds-seed0 \
  --title "Krum vs Multi-Krum: clean and omniscient attack" \
  --output results/krum-vs-multi-krum-omniscient-seed0.png
```

### 多随机种子均值与标准差

完成 seed 0、1、2 后加入 `--aggregate-seeds`。绘图器会按照相同实验配置分组，绘制
均值曲线和正负一个标准差的阴影。

全部六组标准实验：

```bash
python plot_results.py \
  results/*mean-clean-6000rounds-seed* \
  results/*krum-clean-f3-6000rounds-seed* \
  results/*multi-krum-clean-f3-m7-6000rounds-seed* \
  results/*mean-gaussian-f3-fixed-6000rounds-seed* \
  results/*krum-gaussian-f3-fixed-6000rounds-seed* \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed* \
  --aggregate-seeds \
  --title "Three-seed comparison: clean vs Gaussian attack" \
  --output results/all-six-experiments-three-seeds.png
```

只汇总 Krum 和 Multi-Krum 的攻击实验：

```bash
python plot_results.py \
  results/*krum-gaussian-f3-fixed-6000rounds-seed* \
  results/*multi-krum-gaussian-f3-fixed-m7-6000rounds-seed* \
  --aggregate-seeds \
  --title "Krum vs Multi-Krum under Gaussian attack, mean and std" \
  --output results/krum-vs-multi-krum-gaussian-three-seeds.png
```

## 当前参考结果

Mean 无攻击的历史实验在 round 3900 达到最佳测试准确率 72.71%。6000 rounds 时
准确率为 71.95%；继续训练到 10000 rounds 没有提高准确率，并且 test loss 明显
上升。因此标准实验使用 6000 global rounds，并比较完整曲线而不是只看最终点。

已完成的 Multi-Krum + Gaussian fixed 实验使用 `krum_f=3, m=7`：

```text
最佳 Test Accuracy: 70.30%（round 5100）
最终 Test Accuracy: 69.98%
最终 Test Error: 30.02%
选中恶意梯度: 0 / 42000
```

此前部分 Mean 和 Krum 攻击实验采用逐 round 随机恶意 worker，Krum 无攻击历史实验
使用 `krum_f=0`。这些结果仍可作为探索性参考，但不能与新的 fixed、`krum_f=3`
标准矩阵进行严格配对比较。

## 后续计划

1. 补齐 fixed Byzantine worker 下的标准六组实验
2. 使用 seed 0、1、2 重复核心实验并汇总均值与标准差
3. 运行并评估 omniscient Byzantine attack
4. 增加自动批量实验脚本并整理最终实验结论
