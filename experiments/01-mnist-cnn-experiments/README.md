# MNIST 实验速查

该目录承接 MNIST 主线，覆盖从最小 `MLP baseline` 到改进版 `CNN improved` 的完整图像分类训练链路。

## 关联笔记

- [01-MLP与MNIST：从数据预处理到最小分类训练](../../notes/01-MLP与MNIST：从数据预处理到最小分类训练.md)
- [02-CNN与MNIST：从卷积直觉到图像分类升级](../../notes/02-CNN与MNIST：从卷积直觉到图像分类升级.md)

## 实验内容

| 实验 | 作用 | 最佳测试准确率 |
| --- | --- | ---: |
| `MLP baseline` | 建立最小分类训练基线 | `96.12%` |
| `CNN improved` | 观察卷积结构和训练策略带来的提升 | `99.47%` |

## 代表结果

预测可视化用于快速确认改进版 CNN 是否已经稳定学到数字形状特征。

<p align="center">
  <img src="../../assets/showcase/mnist-cnn-predictions.png" alt="MNIST CNN 预测结果" width="920" />
</p>

## 运行命令

```bash
pip install -r ../requirements.txt
python train_mlp.py
python train_cnn.py
```

常用覆盖参数：

```bash
python train_cnn.py --epochs 5 --batch-size 128 --experiment-name cnn-dev
python train_mlp.py --epochs 3 --output-dir outputs/dev-runs
```

## 输出目录

- `data/`：下载的数据集
- `outputs/<experiment-name>/`：配置、指标、最佳权重和预测可视化
- 这些目录默认仅用于本地运行，不纳入版本控制。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_mlp.py` | MLP 训练入口 |
| `train_cnn.py` | CNN 训练入口 |
| `mnist_experiments/models.py` | 模型定义 |
| `mnist_experiments/runner.py` | 训练主流程 |
| `mnist_experiments/data.py` | 数据与数据加载 |
