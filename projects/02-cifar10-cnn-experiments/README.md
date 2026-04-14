# CIFAR-10 实验速查

主阅读入口：

- [03-CIFAR-10与ResNet：从简单CNN到残差网络](../../notes/03-CIFAR-10与ResNet：从简单CNN到残差网络.md)

## 包含实验

| 实验 | 作用 | 最佳测试准确率 |
| --- | --- | ---: |
| `baseline` | 保留最清晰的简单 CNN 对照组 | `73.25%` |
| `improved` | 在 CNN 路线上补齐增强和训练策略 | `87.35%` |
| `resnet` | 切到残差网络继续拉高上限 | `95.33%` |

![CIFAR-10 ResNet predictions](../../assets/showcase/cifar10-resnet-predictions.png)

## 运行命令

```bash
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```

常用覆盖参数：

```bash
python train_baseline.py --epochs 10 --batch-size 256 --experiment-name cifar10-baseline-dev
python train_improved.py --epochs 20 --batch-size 128 --experiment-name cifar10-improved-dev
python train_resnet.py --epochs 30 --batch-size 128 --experiment-name cifar10-resnet-dev
```

## 输出目录

- `data/`：本地下载的数据集
- `outputs/<experiment-name>/`：配置、指标、最佳权重和预测可视化

这些目录默认只作为本地运行目录使用。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_baseline.py` | 简单 CNN 训练入口 |
| `train_improved.py` | 改进版 CNN 训练入口 |
| `train_resnet.py` | ResNet 训练入口 |
| `cifar10_experiments/models.py` | 模型定义 |
| `cifar10_experiments/runner.py` | 训练主流程 |
