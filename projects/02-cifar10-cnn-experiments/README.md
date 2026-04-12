# CIFAR-10 CNN 实验

这个项目围绕 CIFAR-10 做了三版图像分类实验：简单 CNN、改进版 CNN 和 ResNet。

## 项目定位

- `baseline`：简单 CNN，对应最清晰的对照组。
- `improved`：保留 CNN 路线，但补齐 BatchNorm、Dropout、AdamW、CosineAnnealing、标签平滑和数据增强。
- `resnet`：切换到更适合 CIFAR-10 的残差网络路线，继续提升上限。

## 核心结果

| 版本 | 轮数 | 批大小 | 最佳测试准确率 | 最终测试损失 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline` | 20 | 128 | `73.25%` | `1.1329` | 简单 CNN，无增强 |
| `improved` | 30 | 256 | `87.35%` | `0.8294` | 更深 CNN + 增强 + AdamW + Cosine + AMP |
| `resnet` | 100 | 128 | `95.33%` | `0.6180` | ResNet + SGD + MultiStep + AMP + RandomErasing |

- `improved` 相比 `baseline` 提升 `14.10` 个百分点。
- `resnet` 相比 `baseline` 提升 `22.08` 个百分点。
- 从结果上可以比较清楚地看出结构和训练策略的影响。

## 精选展示

![CIFAR-10 ResNet predictions](../../assets/showcase/cifar10-resnet-predictions.png)

这张图保留了一次典型预测结果。

## 如何运行

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
python train_resnet.py --epochs 30 --batch-size 128 --experiment-name cifar10-resnet-dev
```

运行后会在本项目目录下自动生成：

- `data/`：本地下载的数据集
- `outputs/<experiment-name>/`：配置、指标、最佳权重和预测可视化

这些目录默认只作为本地运行目录使用。

## 代码结构

```text
02-cifar10-cnn-experiments/
├─ cifar10_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ utils.py
│  └─ visualize.py
├─ train_baseline.py
├─ train_improved.py
└─ train_resnet.py
```

- 三个 `train_*.py` 负责切换不同实验版本。
- `cifar10_experiments/` 负责训练逻辑、配置管理和结果落盘。

## 延伸阅读

- 总览对比见：[实验结果总览](../../docs/实验结果总览.md)
- 如果想从更基础的图像分类理解开始，建议先看：[MNIST 实验](../01-mnist-cnn-experiments/README.md)
