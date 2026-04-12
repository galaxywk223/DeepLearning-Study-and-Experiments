# MNIST Experiments

这个项目使用 PyTorch 在 MNIST 手写数字识别任务上实现了两个版本：

- `MLP baseline`：单隐藏层全连接网络，作为最小可用基线
- `CNN improved`：两层卷积网络，加入 BatchNorm、Dropout、AdamW 和学习率调度

## 实验结果

以下结果来自本地 CUDA 环境的最新一次完整运行：

| Model        | Epochs | Batch Size | Augmentation      | Best Test Accuracy | Final Test Loss |
| ------------ | -----: | ---------: | ----------------- | -----------------: | --------------: |
| MLP baseline |     10 |         64 | None              |             96.12% |          0.1337 |
| CNN improved |     15 |         64 | RandomRotation(5) |             99.47% |          0.0192 |

对比结论：

- CNN 相比 MLP 提升了 `3.35` 个百分点
- CNN 在第 `12` 个 epoch 达到最佳精度 `99.47%`
- 这个项目清晰展示了从简单基线到改进模型的迭代过程

## 快速运行

```bash
pip install -r ../requirements.txt
python train_mlp.py
python train_cnn.py
```

也可以覆盖默认参数：

```bash
python train_cnn.py --epochs 5 --batch-size 128 --experiment-name cnn-dev
python train_mlp.py --epochs 3 --output-dir outputs/dev-runs
```

## 输出文件

每次运行都会写入 `outputs/<experiment-name>/`：

- `config.json`：本次实验配置
- `metrics.json`：训练历史、最佳精度、最终精度
- `best_model.pt`：最佳 checkpoint
- `predictions.png`：预测结果示例图

当前已生成的实验产物：

- `outputs/mlp-baseline/`
- `outputs/cnn-improved/`

说明：

- `data/` 和 `outputs/` 都是本地运行时生成目录，默认不提交到仓库
- 共享依赖文件位于 `../requirements.txt`

## 项目结构

```text
01-mnist-cnn-experiments/
├─ data/                 # local, gitignored
├─ mnist_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ utils.py
│  └─ visualize.py
├─ outputs/              # local, gitignored
├─ cnn_model.py
├─ train_cnn.py
├─ train_mlp.py
└─ ../requirements.txt   # shared dependency file
```

源码职责：

- `train_mlp.py` / `train_cnn.py`：命令行入口
- `mnist_experiments/models.py`：MLP 和 CNN 模型定义
- `mnist_experiments/data.py`：MNIST 数据集和预处理
- `mnist_experiments/engine.py`：训练与评估循环
- `mnist_experiments/runner.py`：实验编排、checkpoint 保存、metrics 输出
- `mnist_experiments/visualize.py`：预测结果图保存

## 默认配置

`MLP baseline`

- 优化器：`SGD`
- 学习率：`0.01`
- 训练轮数：`10`
- 隐藏层维度：`128`

`CNN improved`

- 优化器：`AdamW`
- 学习率：`0.001`
- 权重衰减：`1e-4`
- 训练轮数：`15`
- 批大小：`64`
- 数据增强：`RandomRotation(5)`
- 隐藏层维度：`512`
- Dropout：`0.5`
- 学习率调度：`ReduceLROnPlateau`

## 工程说明

- 数据默认下载到 `./data`
- 训练集可选随机旋转增强，测试集固定只做标准化，避免评估口径污染
- 模型最佳权重和训练指标统一保存在 `outputs/` 下，便于后续补实验表和可视化
- `cnn_model.py` 当前仅保留兼容导出，核心实现已迁移到 `mnist_experiments/`
