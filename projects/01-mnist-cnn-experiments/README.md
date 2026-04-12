# MNIST 实验

这个项目用 MNIST 做了一条比较基础的实验线：先是 MLP，再是 CNN。

## 项目定位

- `MLP baseline`：单隐藏层全连接网络。
- `CNN improved`：两层卷积网络，加入 BatchNorm、Dropout、AdamW 和学习率调度。
- 两个版本放在一起，方便直接比较。

## 核心结果

| 模型 | 轮数 | 批大小 | 数据增强 | 最佳测试准确率 | 最终测试损失 |
| --- | ---: | ---: | --- | ---: | ---: |
| `MLP baseline` | 10 | 64 | 无 | `96.12%` | `0.1337` |
| `CNN improved` | 15 | 64 | `RandomRotation(5)` | `99.47%` | `0.0192` |

- `CNN improved` 比 `MLP baseline` 高 `3.35` 个百分点。
- 这一组结果适合拿来看从全连接到卷积网络的差别。

## 精选展示

![MNIST CNN predictions](../../assets/showcase/mnist-cnn-predictions.png)

这张图保留了一次典型预测结果。

## 如何运行

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

运行后会在本项目目录下自动生成：

- `data/`：下载的数据集
- `outputs/<experiment-name>/`：配置、指标、最佳权重和预测可视化

这些目录默认仅用于本地运行，不纳入版本控制。

## 代码结构

```text
01-mnist-cnn-experiments/
├─ mnist_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ utils.py
│  └─ visualize.py
├─ cnn_model.py
├─ train_cnn.py
└─ train_mlp.py
```

- `train_mlp.py` 和 `train_cnn.py` 是命令行入口。
- `mnist_experiments/` 负责数据、配置、训练、可视化和结果落盘。

## 延伸阅读

- 原理与实现展开见：[01-MLP与MNIST：从数据预处理到训练流程](../../notes/01-MLP与MNIST：从数据预处理到训练流程.md)
- 卷积相关内容见：[02-CNN与MNIST：从卷积直觉到图像分类实现](../../notes/02-CNN与MNIST：从卷积直觉到图像分类实现.md)
