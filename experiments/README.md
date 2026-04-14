# 实验索引

`experiments/` 负责保存可运行实验和最小运行说明，建议先从 [notes/README.md](../notes/README.md) 建立概念，再回到这里选具体实验运行。

## 当前实验

| 目录 | 主笔记 | 主入口 | 说明 |
| --- | --- | --- | --- |
| [01-mnist-cnn-experiments](./01-mnist-cnn-experiments/README.md) | [01-MLP 与 MNIST](../notes/01-MLP与MNIST：从数据预处理到最小分类训练.md)、[02-CNN 与 MNIST](../notes/02-CNN与MNIST：从卷积直觉到图像分类升级.md) | `python train_cnn.py` | 从 MLP 到 CNN 的 MNIST 分类主线 |
| [02-cifar10-cnn-experiments](./02-cifar10-cnn-experiments/README.md) | [03-CIFAR-10 与 ResNet](../notes/03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | `python train_resnet.py` | 从简单 CNN 升级到 ResNet 的 CIFAR-10 实验 |
| [03-char-transformer-experiments](./03-char-transformer-experiments/README.md) | [04-自注意力机制](../notes/04-自注意力机制：从Q、K、V到缩放点积注意力.md)、[05-Transformer 语言模型](../notes/05-Transformer语言模型：从位置编码到最小可训练实现.md) | `python train_transformer.py` | 最小字符级 Transformer 训练与采样 |
| [04-subword-gpt-experiments](./04-subword-gpt-experiments/README.md) | [06-子词级 GPT](../notes/06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | `python train_gpt.py` | 子词级 GPT 训练、采样与 tokenizer 工作流 |

## 运行前准备

共享依赖位于 [requirements.txt](./requirements.txt)。常用入口如下：

```bash
pip install -r experiments/requirements.txt
cd experiments/01-mnist-cnn-experiments
python train_cnn.py
```

```bash
cd experiments/04-subword-gpt-experiments
python train_gpt.py
```

每个实验目录都会把本地数据写入 `data/`，把模型和指标写入 `outputs/`。
