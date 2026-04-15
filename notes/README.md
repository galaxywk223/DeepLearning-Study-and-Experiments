# 学习笔记

`notes/` 是这个仓库的主阅读层。仓库总览见根目录 [README](../README.md)，章节阅读顺序如下。

## 推荐阅读顺序

| 章节 | 主题 | 章节作用 | 实验入口 |
| --- | --- | --- | --- |
| [01-MLP与MNIST：从数据预处理到最小分类训练](./01-MLP与MNIST：从数据预处理到最小分类训练.md) | MLP 与 MNIST | 最小分类训练链路建立 | [MNIST 实验速查](../experiments/01-mnist-cnn-experiments/README.md) |
| [02-CNN与MNIST：从卷积直觉到图像分类升级](./02-CNN与MNIST：从卷积直觉到图像分类升级.md) | CNN 与 MNIST | 卷积直觉与图像建模差异说明 | [MNIST 实验速查](../experiments/01-mnist-cnn-experiments/README.md) |
| [03-CIFAR-10与ResNet：从简单CNN到残差网络](./03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | CIFAR-10 与 ResNet | 图像分类工程化升级 | [CIFAR-10 实验速查](../experiments/02-cifar10-cnn-experiments/README.md) |
| [04-自注意力机制：从Q、K、V到缩放点积注意力](./04-自注意力机制：从Q、K、V到缩放点积注意力.md) | 自注意力机制 | Transformer 前置注意力直觉建立 | [字符级 Transformer 实验速查](../experiments/03-char-transformer-experiments/README.md) |
| [05-Transformer语言模型：从位置编码到最小可训练实现](./05-Transformer语言模型：从位置编码到最小可训练实现.md) | Transformer 语言模型 | 最小可训练 Transformer 实现 | [字符级 Transformer 实验速查](../experiments/03-char-transformer-experiments/README.md) |
| [06-子词级GPT：从BPE到更像真实LLM的训练流程](./06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | 子词级 GPT | 子词级 tokenizer 与采样流程补全 | [子词级 GPT 实验速查](../experiments/04-subword-gpt-experiments/README.md) |
| [07-指令微调与LoRA：从预训练模型到领域助教](./07-指令微调与LoRA：从预训练模型到领域助教.md) | 指令微调与 LoRA | 真实 LLM 轻量微调流程衔接 | [Notes Assistant SFT 实验速查](../experiments/05-notes-assistant-sft-experiments/README.md) |

## 与实验对应

| 实验目录 | 对应章节 | 说明 |
| --- | --- | --- |
| [experiments/01-mnist-cnn-experiments](../experiments/01-mnist-cnn-experiments/README.md) | `01`、`02` | 从 MLP 基线到改进版 CNN 的第一条分类主线 |
| [experiments/02-cifar10-cnn-experiments](../experiments/02-cifar10-cnn-experiments/README.md) | `03` | 从简单 CNN 到 ResNet 的 CIFAR-10 分类实验 |
| [experiments/03-char-transformer-experiments](../experiments/03-char-transformer-experiments/README.md) | `04`、`05` | 自注意力到最小字符级语言模型实现 |
| [experiments/04-subword-gpt-experiments](../experiments/04-subword-gpt-experiments/README.md) | `06` | 子词级 tokenizer、训练和采样流程 |
| [experiments/05-notes-assistant-sft-experiments](../experiments/05-notes-assistant-sft-experiments/README.md) | `07` | 基于现有笔记的数据构建、SFT、LoRA 与评测流程 |
