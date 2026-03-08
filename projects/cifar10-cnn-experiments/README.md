# CIFAR-10 CNN Experiments

这个目录现在包含两个阶段：

- `baseline`：简单 CNN 起点
- `improved`：在 baseline 上加入工程化优化版本
- `resnet`：第三版残差网络，继续冲更高上限

## 第一版定位

- 数据集：`CIFAR-10`
- 任务：`32x32` 彩色图像 10 分类
- 模型：两层卷积 + ReLU + MaxPool + 两层全连接
- 训练策略：`Adam` + `CrossEntropyLoss`
- 数据处理：只做 `ToTensor + Normalize`
- 不包含的优化：数据增强、BN、Dropout、残差结构、AMP、标签平滑

这版的目的不是追求最高精度，而是建立一个清晰的起点，复现“简单 CNN 在 CIFAR-10 上准确率有限，因此需要进一步工程化优化”的项目背景。

## 第二版定位

第二版继续保留 CNN 路线，但补上更接近实际项目的训练优化：

- 更深的卷积结构
- BatchNorm + Dropout
- `AdamW`
- `CosineAnnealingLR`
- `Label Smoothing`
- CUDA 下可选 `AMP`
- 数据增强：`RandomCrop`、`RandomHorizontalFlip`、`ColorJitter`

目标是把 baseline 从“能跑通”推进到“有明显可展示的优化幅度”。

## 第三版定位

第三版不再只是“把普通 CNN 继续调强”，而是切换到更适合 CIFAR-10 的残差网络路线：

- `CIFAR-style ResNet`
- `SGD + Nesterov momentum`
- `MultiStepLR`
- `Label Smoothing`
- `AMP`
- 更完整的数据增强，包括 `RandomErasing`

目标是继续把准确率往 `90%+` 推进。

## 当前结果

以下结果来自本地 CUDA 环境的实际运行产物：

| Variant  | Epochs | Batch Size | Best Test Accuracy | Final Test Loss | Notes                                            |
| -------- | -----: | ---------: | -----------------: | --------------: | ------------------------------------------------ |
| baseline |     20 |        128 |             73.25% |          1.1329 | 简单 CNN，无增强                                 |
| improved |     30 |        256 |             87.35% |          0.8294 | 更深 CNN + 增强 + AdamW + Cosine + AMP           |
| resnet   |    100 |        128 |             95.33% |          0.6180 | 残差网络 + SGD + MultiStep + AMP + RandomErasing |

对比结论：

- improved 相比 baseline 提升了 `14.10` 个百分点
- resnet 相比 baseline 提升了 `22.08` 个百分点
- resnet 相比 improved 进一步提升了 `7.98` 个百分点
- baseline 在约第 `4-5` 个 epoch 后接近瓶颈，并出现明显过拟合迹象
- improved 在第 `27` 个 epoch 达到当前最佳精度 `87.35%`
- resnet 在第 `92` 个 epoch 达到当前最佳精度 `95.33%`
- 这个项目清晰展示了“baseline 表现有限，通过工程化优化和结构升级显著提升性能”的迭代路径，并且已经超过了 `90%+` 目标

## 运行

```bash
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```

也可以覆盖默认参数：

```bash
python train_baseline.py --epochs 10 --batch-size 256 --experiment-name cifar10-baseline-dev
python train_improved.py --epochs 20 --batch-size 128 --experiment-name cifar10-improved-dev
python train_resnet.py --batch-size 128 --num-workers 4 --use-amp
```

Windows + CUDA 建议：

- 优先从 `--num-workers 2` 或 `--num-workers 4` 开始，不建议一上来设到 `8`
- 如果训练打印完 `Running ...` 后长时间不进入 `Epoch 01/...`，先试 `--num-workers 0`
- `improved` 版在 CUDA 下建议加 `--use-amp`

## 输出文件

每次运行都会写入 `outputs/<experiment-name>/`：

- `config.json`：本次实验配置
- `metrics.json`：训练历史、最佳精度、最终精度
- `best_model.pt`：最佳 checkpoint
- `predictions.png`：预测结果示例图
- `metrics.json` 中也会记录实际生效的 DataLoader 配置

说明：

- `data/` 和 `outputs/` 都是本地运行时生成目录，默认不提交到仓库
- 共享依赖文件位于 `../requirements.txt`

## 项目结构

```text
cifar10-cnn-experiments/
├─ data/                 # local, gitignored
├─ cifar10_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ utils.py
│  └─ visualize.py
├─ outputs/              # local, gitignored
├─ train_baseline.py
├─ train_improved.py
├─ train_resnet.py
└─ ../requirements.txt   # shared dependency file
```

## 源码职责

- `train_baseline.py`：命令行入口
- `train_improved.py`：优化版训练入口
- `train_resnet.py`：第三版残差网络入口
- `cifar10_experiments/models.py`：baseline / improved / resnet 模型定义
- `cifar10_experiments/data.py`：CIFAR-10 数据集与预处理
- `cifar10_experiments/engine.py`：训练与评估循环
- `cifar10_experiments/runner.py`：实验编排、checkpoint 保存、metrics 输出
- `cifar10_experiments/visualize.py`：预测结果图保存

## 默认配置

`baseline`

- Optimizer: `Adam`
- Learning rate: `0.001`
- Epochs: `20`
- Batch size: `128`
- Num workers: `2`

`improved`

- Optimizer: `AdamW`
- Learning rate: `0.0003`
- Weight decay: `5e-4`
- Epochs: `30`
- Batch size: `128`
- Num workers: `4`
- Label smoothing: `0.1`
- Scheduler: `CosineAnnealingLR`
- AMP: enabled on CUDA
- Augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + `ColorJitter`
- Windows 下会自动将 `num_workers` 上限控制在 `4`

`resnet`

- Optimizer: `SGD` with Nesterov momentum
- Learning rate: `0.1`
- Weight decay: `5e-4`
- Epochs: `100`
- Batch size: `128`
- Num workers: `4`
- Label smoothing: `0.1`
- Scheduler: `MultiStepLR`
- AMP: enabled on CUDA
- Augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + `ColorJitter` + `RandomErasing`
- 当前结果：`95.33%`
