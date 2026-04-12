# 学习笔记索引

这里收录的是仓库里的详细教程型内容。

如果项目页解决的是“我做了什么、结果怎样、怎么运行”，那么这里解决的是“我为什么这么做、背后的原理是什么、代码应该怎样理解”。

## 推荐阅读方式

- 想快速浏览仓库：先看根目录 [README](../README.md)。
- 想理解项目之间的路线关系：看 [学习路线与项目导航](../docs/学习路线与项目导航.md)。
- 想系统读原理：按下面的阶段顺序往下看。

## 阶段一：从图像分类入门

- [01-MLP与MNIST：从数据预处理到训练流程.md](./01-MLP与MNIST：从数据预处理到训练流程.md)
  适合第一次进入这个仓库时阅读。
  对应项目：[MNIST 实验](../projects/01-mnist-cnn-experiments/README.md)

- [02-CNN与MNIST：从卷积直觉到图像分类实现.md](./02-CNN与MNIST：从卷积直觉到图像分类实现.md)
  适合已经理解 MLP，希望进一步理解卷积为什么更适合图像任务时阅读。
  对应项目：[MNIST 实验](../projects/01-mnist-cnn-experiments/README.md)

## 阶段二：进入 Transformer

- [03-自注意力机制：从Q、K、V到缩放点积注意力.md](./03-自注意力机制：从Q、K、V到缩放点积注意力.md)
  适合准备进入语言模型之前阅读，用来建立自注意力的核心直觉。
  对应项目：[字符级 Transformer 实验](../projects/03-char-transformer-experiments/README.md)

- [04-Transformer语言模型：从位置编码到最小可训练实现.md](./04-Transformer语言模型：从位置编码到最小可训练实现.md)
  适合想把“注意力公式”真正落到最小可训练语言模型代码中的读者。
  对应项目：[字符级 Transformer 实验](../projects/03-char-transformer-experiments/README.md)

## 阶段三：从最小模型走向更真实的 GPT 工作流

- [05-子词级GPT：从BPE到更像真实LLM的训练流程.md](./05-子词级GPT：从BPE到更像真实LLM的训练流程.md)
  适合已经理解最小 Transformer，希望继续理解 tokenizer、special tokens 和 padding mask 时阅读。
  对应项目：[子词级 GPT 实验](../projects/04-subword-gpt-experiments/README.md)

## 说明

- 这些笔记以“直觉理解 + 数学解释 + 代码落地”为主。
- 图片资源统一存放在 `../assets/images/`。
- 项目页负责展示结果，笔记页负责展开细节，因此这里不再承担仓库导航职责。
