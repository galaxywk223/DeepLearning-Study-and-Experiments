# 深度学习学习与实验

这个仓库整理了深度学习方向的中文学习笔记和配套实验，当前主线覆盖图像分类、语言模型，以及基于预训练模型的轻量微调实验。主阅读层放在 `notes/`，可运行实验统一放在 `experiments/`，代码实现以 PyTorch 为主。

## 仓库导航

- [notes/README.md](notes/README.md)：章节顺序、主线概览和索引入口。
- [experiments/README.md](experiments/README.md)：实验索引、运行入口和目录速查。
- [assets/showcase/](assets/showcase/)：根 README 和正文中直接引用的代表结果图。

## 学习主线

| 章节 | 主题 | 主要内容 | 实验入口 |
| --- | --- | --- | --- |
| [01](./notes/01-MLP与MNIST：从数据预处理到最小分类训练.md) | MLP 与 MNIST | 从数据预处理到最小分类训练链路，建立第一条图像分类基线 | [MNIST 实验速查](./experiments/01-mnist-cnn-experiments/README.md) |
| [02](./notes/02-CNN与MNIST：从卷积直觉到图像分类升级.md) | CNN 与 MNIST | 从全连接升级到卷积网络，理解局部感受野和参数共享 | [MNIST 实验速查](./experiments/01-mnist-cnn-experiments/README.md) |
| [03](./notes/03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | CIFAR-10 与 ResNet | 从简单 CNN 走向更真实的图像分类工程化 | [CIFAR-10 实验速查](./experiments/02-cifar10-cnn-experiments/README.md) |
| [04](./notes/04-自注意力机制：从Q、K、V到缩放点积注意力.md) | 自注意力机制 | 建立 `Q / K / V` 和缩放点积注意力直觉 | [字符级 Transformer 实验速查](./experiments/03-char-transformer-experiments/README.md) |
| [05](./notes/05-Transformer语言模型：从位置编码到最小可训练实现.md) | Transformer 语言模型 | 把注意力落成最小 decoder-only 语言模型 | [字符级 Transformer 实验速查](./experiments/03-char-transformer-experiments/README.md) |
| [06](./notes/06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | 子词级 GPT | 补齐 tokenizer、padding 和采样控制，走向更完整的 GPT 工作流 | [子词级 GPT 实验速查](./experiments/04-subword-gpt-experiments/README.md) |
| [07](./notes/07-指令微调与LoRA：从预训练模型到领域助教.md) | 指令微调与 LoRA | 从预训练模型出发补齐 `SFT + LoRA/QLoRA + 评测 + Demo` 的轻量微调流程 | [Notes Assistant SFT 实验速查](./experiments/05-notes-assistant-sft-experiments/README.md) |

## 结果速览

| 主线 | 代表结果 | 主要看点 |
| --- | --- | --- |
| MNIST | `CNN improved` 测试集准确率 `99.47%` | 从 MLP 到 CNN 的第一条完整分类训练线 |
| CIFAR-10 | `ResNet` 测试集准确率 `95.33%` | 结构升级和训练策略如何一起拉高上限 |
| Character Transformer | `transformer v3` 验证集困惑度 `4.63` | 最小字符级 Transformer 的实现与生成表现 |
| Subword GPT | `subword-gpt v2` 验证集困惑度 `13.19` | 更接近真实 GPT 的 tokenizer 和训练流程 |
| Notes Assistant SFT | 平均字符级 F1 `0.285 -> 0.444` | 基于 `Qwen2.5-0.5B-Instruct + LoRA` 的领域助教在 `30` 道 held-out 题上有 `93.33%` 样本优于基座 |

语言模型两条结果不能直接横向比较。token 粒度不同，比较应分别放在各自主线内部进行。

第 `07` 章对应的是基于预训练模型的轻量微调主线，重点放在指令数据、adapter 训练和评测流程。

## 精选展示

### CIFAR-10 / ResNet

图像分类主线里，`ResNet` 是目前最能体现工程化训练差异的一组结果。

<p align="center">
  <img src="./assets/showcase/cifar10-resnet-predictions.png" alt="CIFAR-10 ResNet 预测结果" width="920" />
</p>

### Character Transformer v3

字符级语言模型的收敛曲线用于观察最小 Transformer 骨架何时开始稳定工作。

<p align="center">
  <img src="./assets/showcase/char-transformer-v3-loss-curve.png" alt="Character Transformer v3 收敛曲线" width="760" />
</p>

### Subword GPT v2

子词级 GPT 在更完整的 tokenizer 和采样流程下，验证困惑度明显继续下降。

<p align="center">
  <img src="./assets/showcase/subword-gpt-v2-loss-curve.png" alt="Subword GPT v2 收敛曲线" width="760" />
</p>

### Notes Assistant SFT

第 `07` 章笔记中保留了基座模型与微调后模型的代表问答对照。下图仅展示对应的缩略图，完整说明见 [07-指令微调与LoRA：从预训练模型到领域助教](./notes/07-指令微调与LoRA：从预训练模型到领域助教.md)。

<p align="center">
  <img src="./assets/showcase/notes-assistant-qwen25-0p5b-public-results.png" alt="Notes Assistant 公开代表结果" width="920" />
</p>

## 快速开始

共享依赖位于 `experiments/requirements.txt`。常用运行入口如下：

```bash
pip install -r experiments/requirements.txt
cd experiments/01-mnist-cnn-experiments
python train_cnn.py
```

```bash
cd experiments/03-char-transformer-experiments
python train_transformer.py
```

```bash
cd experiments/05-notes-assistant-sft-experiments
python prepare_dataset.py --overwrite
python train_sft.py --smoke
```

运行后，各实验会在自己的目录下生成：

- `data/`：数据集、语料或 tokenizer 文件
- `outputs/<experiment-name>/`：配置、指标、最佳权重、图表和采样结果

这些目录默认只用于本地运行，不纳入版本控制。

## 仓库结构

```text
DeepLearning-Study-and-Experiments/
├─ assets/
│  ├─ images/
│  └─ showcase/
├─ notes/
│  ├─ README.md
│  ├─ 01-MLP与MNIST：从数据预处理到最小分类训练.md
│  ├─ 02-CNN与MNIST：从卷积直觉到图像分类升级.md
│  ├─ 03-CIFAR-10与ResNet：从简单CNN到残差网络.md
│  ├─ 04-自注意力机制：从Q、K、V到缩放点积注意力.md
│  ├─ 05-Transformer语言模型：从位置编码到最小可训练实现.md
│  ├─ 06-子词级GPT：从BPE到更像真实LLM的训练流程.md
│  └─ 07-指令微调与LoRA：从预训练模型到领域助教.md
├─ experiments/
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ 01-mnist-cnn-experiments/
│  ├─ 02-cifar10-cnn-experiments/
│  ├─ 03-char-transformer-experiments/
│  ├─ 04-subword-gpt-experiments/
│  └─ 05-notes-assistant-sft-experiments/
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 开源协议

本仓库中的代码、笔记与文档内容基于 [MIT License](./LICENSE) 开源。第三方数据集、论文内容、模型思想及其他原始资料，不因本仓库采用 MIT 协议而自动转授额外权利。
