# 章节索引

`notes/` 是这个仓库的主阅读层。每一章都尽量把原理、实验结论、运行方式和代码入口放在一起，项目页只保留速查信息。

如果你刚进入这个仓库，建议先看根目录 [README](../README.md)，再按下面的顺序往下读。

## 推荐顺序

| 章节 | 主题 | 适合什么时候读 | 对应项目速查 |
| --- | --- | --- | --- |
| [01-MLP与MNIST：从数据预处理到最小分类训练](./01-MLP与MNIST：从数据预处理到最小分类训练.md) | MLP 与 MNIST | 想先跑通第一条最小分类训练链路时 | [MNIST 实验速查](../projects/01-mnist-cnn-experiments/README.md) |
| [02-CNN与MNIST：从卷积直觉到图像分类升级](./02-CNN与MNIST：从卷积直觉到图像分类升级.md) | CNN 与 MNIST | 想理解卷积为什么比 MLP 更适合图像时 | [MNIST 实验速查](../projects/01-mnist-cnn-experiments/README.md) |
| [03-CIFAR-10与ResNet：从简单CNN到残差网络](./03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | CIFAR-10 与 ResNet | 想从入门分类走向更真实的图像分类工程化时 | [CIFAR-10 实验速查](../projects/02-cifar10-cnn-experiments/README.md) |
| [04-自注意力机制：从Q、K、V到缩放点积注意力](./04-自注意力机制：从Q、K、V到缩放点积注意力.md) | 自注意力机制 | 准备进入 Transformer 之前 | [字符级 Transformer 实验速查](../projects/03-char-transformer-experiments/README.md) |
| [05-Transformer语言模型：从位置编码到最小可训练实现](./05-Transformer语言模型：从位置编码到最小可训练实现.md) | Transformer 语言模型 | 想把注意力真正落到可训练代码里时 | [字符级 Transformer 实验速查](../projects/03-char-transformer-experiments/README.md) |
| [06-子词级GPT：从BPE到更像真实LLM的训练流程](./06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | 子词级 GPT | 想继续补 tokenizer、padding 和采样控制时 | [子词级 GPT 实验速查](../projects/04-subword-gpt-experiments/README.md) |

## 阅读提示

- 每篇主笔记都尽量先交代“这章做什么、对应哪个实验、结果怎样”，再进入原理展开。
- 如果你只想运行代码，直接跳到对应项目速查页即可。
- 图片资源统一放在 `../assets/images/` 和 `../assets/showcase/`。
