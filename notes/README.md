# 学习笔记

`notes/` 是这个仓库的主阅读层。每一章都尽量把原理、实验结论、运行方式和代码入口放在一起，建议先看根目录 [README](../README.md)，再按下面的顺序往下读。

## 推荐阅读顺序

| 章节 | 主题 | 适合什么时候读 | 实验入口 |
| --- | --- | --- | --- |
| [01-MLP与MNIST：从数据预处理到最小分类训练](./01-MLP与MNIST：从数据预处理到最小分类训练.md) | MLP 与 MNIST | 想先跑通第一条最小分类训练链路时 | [MNIST 实验速查](../experiments/01-mnist-cnn-experiments/README.md) |
| [02-CNN与MNIST：从卷积直觉到图像分类升级](./02-CNN与MNIST：从卷积直觉到图像分类升级.md) | CNN 与 MNIST | 想理解卷积为什么比 MLP 更适合图像时 | [MNIST 实验速查](../experiments/01-mnist-cnn-experiments/README.md) |
| [03-CIFAR-10与ResNet：从简单CNN到残差网络](./03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | CIFAR-10 与 ResNet | 想从入门分类走向更真实的图像分类工程化时 | [CIFAR-10 实验速查](../experiments/02-cifar10-cnn-experiments/README.md) |
| [04-自注意力机制：从Q、K、V到缩放点积注意力](./04-自注意力机制：从Q、K、V到缩放点积注意力.md) | 自注意力机制 | 准备进入 Transformer 之前 | [字符级 Transformer 实验速查](../experiments/03-char-transformer-experiments/README.md) |
| [05-Transformer语言模型：从位置编码到最小可训练实现](./05-Transformer语言模型：从位置编码到最小可训练实现.md) | Transformer 语言模型 | 想把注意力真正落到可训练代码里时 | [字符级 Transformer 实验速查](../experiments/03-char-transformer-experiments/README.md) |
| [06-子词级GPT：从BPE到更像真实LLM的训练流程](./06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | 子词级 GPT | 想继续补 tokenizer、padding 和采样控制时 | [子词级 GPT 实验速查](../experiments/04-subword-gpt-experiments/README.md) |
| [07-指令微调与LoRA：从预训练模型到领域助教](./07-指令微调与LoRA：从预训练模型到领域助教.md) | 指令微调与 LoRA | 准备从最小 GPT 继续走向真实 LLM 微调流程时 | [Notes Assistant SFT 实验速查](../experiments/05-notes-assistant-sft-experiments/README.md) |

## 与实验对应

| 实验目录 | 对应章节 | 说明 |
| --- | --- | --- |
| [experiments/01-mnist-cnn-experiments](../experiments/01-mnist-cnn-experiments/README.md) | `01`、`02` | 从 MLP 基线到改进版 CNN 的第一条分类主线 |
| [experiments/02-cifar10-cnn-experiments](../experiments/02-cifar10-cnn-experiments/README.md) | `03` | 从简单 CNN 到 ResNet 的 CIFAR-10 分类实验 |
| [experiments/03-char-transformer-experiments](../experiments/03-char-transformer-experiments/README.md) | `04`、`05` | 自注意力到最小字符级语言模型实现 |
| [experiments/04-subword-gpt-experiments](../experiments/04-subword-gpt-experiments/README.md) | `06` | 子词级 tokenizer、训练和采样流程 |
| [experiments/05-notes-assistant-sft-experiments](../experiments/05-notes-assistant-sft-experiments/README.md) | `07` | 基于现有笔记的数据构建、SFT、LoRA 与评测流程 |
