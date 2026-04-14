# CIFAR-10 实验速查

这个目录承接 CIFAR-10 图像分类主线，依次保留 `baseline`、`improved` 和 `resnet` 三组可对照的实验。

## 关联笔记

- [03-CIFAR-10与ResNet：从简单CNN到残差网络](../../notes/03-CIFAR-10与ResNet：从简单CNN到残差网络.md)

## 实验内容

| 实验 | 作用 | 最佳测试准确率 |
| --- | --- | ---: |
| `baseline` | 保留最清晰的简单 CNN 对照组 | `73.25%` |
| `improved` | 在 CNN 路线上补齐增强和训练策略 | `87.35%` |
| `resnet` | 切到残差网络继续拉高上限 | `95.33%` |

## 代表结果

`ResNet` 预测可视化适合对照模型升级后在复杂小图像上的识别表现。

<p align="center">
  <img src="../../assets/showcase/cifar10-resnet-predictions.png" alt="CIFAR-10 ResNet 预测结果" width="920" />
</p>

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
- 这些目录默认只作为本地运行目录使用。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_baseline.py` | 简单 CNN 训练入口 |
| `train_improved.py` | 改进版 CNN 训练入口 |
| `train_resnet.py` | ResNet 训练入口 |
| `cifar10_experiments/models.py` | 模型定义 |
| `cifar10_experiments/runner.py` | 训练主流程 |
