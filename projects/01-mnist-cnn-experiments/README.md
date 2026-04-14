# MNIST 实验速查

主阅读入口：

- [01-MLP与MNIST：从数据预处理到最小分类训练](../../notes/01-MLP与MNIST：从数据预处理到最小分类训练.md)
- [02-CNN与MNIST：从卷积直觉到图像分类升级](../../notes/02-CNN与MNIST：从卷积直觉到图像分类升级.md)

## 包含实验

| 实验 | 作用 | 最佳测试准确率 |
| --- | --- | ---: |
| `MLP baseline` | 建立最小分类训练基线 | `96.12%` |
| `CNN improved` | 观察卷积结构与训练策略带来的提升 | `99.47%` |

![MNIST CNN predictions](../../assets/showcase/mnist-cnn-predictions.png)

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

这些目录默认仅用于本地运行，不纳入版本控制。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_mlp.py` | MLP 训练入口 |
| `train_cnn.py` | CNN 训练入口 |
| `mnist_experiments/models.py` | 模型定义 |
| `mnist_experiments/runner.py` | 训练主流程 |
| `mnist_experiments/data.py` | 数据与数据加载 |
