# DeepLearning 学习笔记与实验

这个仓库包含两个方向的内容：

- 深度学习实验项目：围绕 MNIST、CIFAR-10 和字符级语言模型构建可运行、可对比的 PyTorch 训练代码
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

### Character Transformer Experiments

项目路径：[projects/char-transformer-experiments/README.md](projects/char-transformer-experiments/README.md)

- `bigram`：最小字符级 next-token baseline
- `transformer`：带位置编码、因果掩码、多头注意力和 FFN 的 decoder-only Transformer
- 最新结果：`transformer v3` 验证集 `perplexity = 4.63`
- 重点：把自注意力笔记继续推进到可训练、可生成文本的最小语言模型实现，并支持不同 `temperature` 的生成对比

## 笔记目录

笔记索引见 [notes/README.md](notes/README.md)。

- [notes/mlp-mnist.md](notes/mlp-mnist.md)：MLP 手写数字识别
- [notes/cnn-mnist.md](notes/cnn-mnist.md)：CNN 数学直觉与 MNIST 实战
- [notes/transformer-self-attention.md](notes/transformer-self-attention.md)：Transformer 自注意力机制推导
- [notes/transformer-language-model.md](notes/transformer-language-model.md)：从位置编码到最小 Transformer 语言模型实现

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

### Character Transformer

```bash
cd projects/char-transformer-experiments
pip install -r ../requirements.txt
python train_bigram.py
python train_transformer.py
python generate_samples.py --run-dir outputs/tinyshakespeare-transformer-v3 --temperatures 0.6 0.75 0.9
```

说明：

- 三个实验项目现在都会默认把数据和输出写入各自项目目录
- 仍然可以通过 `--data-dir` 和 `--output-dir` 覆盖默认路径

## 仓库结构

```text
DeepLearning/
├─ notes/
├─ assets/
│  └─ images/
├─ projects/
│  ├─ requirements.txt
│  ├─ mnist-cnn-experiments/
│  ├─ cifar10-cnn-experiments/
│  └─ char-transformer-experiments/
└─ README.md
```

## 开源协议

本仓库中的代码、笔记与文档内容基于 [MIT License](./LICENSE) 开源。

补充说明：

- 本仓库的许可证仅覆盖当前仓库中自行编写和整理的训练代码、学习笔记、图示与文档结构。
- 数据集、论文内容、模型原始思想以及其他第三方原始资料，不因本仓库采用 MIT 协议而自动转授任何额外权利。
