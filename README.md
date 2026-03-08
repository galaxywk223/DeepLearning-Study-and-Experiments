# DeepLearning 学习笔记与实验

这个仓库包含两个方向的内容：

- 深度学习实验项目：围绕 MNIST 和 CIFAR-10 构建可运行、可对比的 PyTorch 训练代码
- 学习笔记：记录模型原理、数学直觉和实现过程

## 项目概览

### MNIST Experiments

项目路径：[projects/mnist-cnn-experiments/README.md](projects/mnist-cnn-experiments/README.md)

- `MLP baseline`：测试集准确率 `96.12%`
- `CNN improved`：测试集准确率 `99.47%`
- 重点：从全连接基线升级到卷积模型，并完成训练流程模块化、结果落盘和可视化输出

### CIFAR-10 CNN Experiments

项目路径：[projects/cifar10-cnn-experiments/README.md](projects/cifar10-cnn-experiments/README.md)

- `baseline`：测试集准确率 `73.25%`
- `improved`：测试集准确率 `87.35%`
- `resnet`：测试集准确率 `95.33%`
- 重点：沿着简单 CNN、工程化优化版、残差网络三条线做出清晰的性能迭代

## 笔记目录

笔记索引见 [notes/README.md](notes/README.md)。

- [notes/mlp-mnist.md](notes/mlp-mnist.md)：MLP 手写数字识别
- [notes/cnn-mnist.md](notes/cnn-mnist.md)：CNN 数学直觉与 MNIST 实战
- [notes/transformer-self-attention.md](notes/transformer-self-attention.md)：Transformer 自注意力机制推导

## 快速开始

共享依赖文件位于 `projects/requirements.txt`。

### MNIST

```bash
cd projects/mnist-cnn-experiments
pip install -r ../requirements.txt
python train_mlp.py
python train_cnn.py
```

### CIFAR-10

```bash
cd projects/cifar10-cnn-experiments
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```

## 仓库结构

```text
DeepLearning/
├─ notes/
├─ assets/
│  └─ images/
├─ projects/
│  ├─ requirements.txt
│  ├─ mnist-cnn-experiments/
│  └─ cifar10-cnn-experiments/
└─ README.md
```
