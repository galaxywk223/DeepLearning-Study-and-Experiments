# DeepLearning 学习笔记与实验

这个仓库整理了学习深度学习过程中的笔记（偏原理推导 + 直觉类比）以及可运行的深度学习实验代码。

## 导航

- 笔记：`notes/`
  - 多层感知机（MLP）手写数字识别：[notes/mlp-mnist.md](notes/mlp-mnist.md)
  - 卷积神经网络（CNN）数学直觉与 MNIST 实战：[notes/cnn-mnist.md](notes/cnn-mnist.md)
  - Transformer 自注意力机制推导：[notes/transformer-self-attention.md](notes/transformer-self-attention.md)
- 代码：
  - `projects/mnist-cnn-experiments/`（见 [projects/mnist-cnn-experiments/README.md](projects/mnist-cnn-experiments/README.md)）
  - `projects/cifar10-cnn-experiments/`（见 [projects/cifar10-cnn-experiments/README.md](projects/cifar10-cnn-experiments/README.md)）
- 资源：`assets/`（图片等）

## 快速运行

共享依赖文件位于 `projects/requirements.txt`。

### MNIST 实验

在 `projects/mnist-cnn-experiments/` 目录下运行：

```bash
cd projects/mnist-cnn-experiments
pip install -r ../requirements.txt
python train_mlp.py
python train_cnn.py
```

### CIFAR-10 实验

在 `projects/cifar10-cnn-experiments/` 目录下运行：

```bash
cd projects/cifar10-cnn-experiments
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```
