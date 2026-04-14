# 深度学习学习笔记与实验

这个仓库现在按“章节主笔记 + 项目速查页”组织。主阅读路径集中在 [`notes/`](./notes/README.md)，实验运行方式和代码入口集中在 `projects/*/README.md`，尽量减少在不同目录之间来回跳转。

## 从哪里开始

- 想按学习路径往下读：从 [章节索引](./notes/README.md) 开始。
- 想快速看实验入口：直接进入各项目速查页。
- 想先看结果：先看下面这张总表，再进入对应章节。

## 学习主线

| 章节 | 主题 | 你会学到什么 | 对应项目速查 |
| --- | --- | --- | --- |
| [01](./notes/01-MLP与MNIST：从数据预处理到最小分类训练.md) | MLP 与 MNIST | 从数据预处理到最小分类训练链路 | [MNIST 实验速查](./projects/01-mnist-cnn-experiments/README.md) |
| [02](./notes/02-CNN与MNIST：从卷积直觉到图像分类升级.md) | CNN 与 MNIST | 从全连接升级到卷积网络 | [MNIST 实验速查](./projects/01-mnist-cnn-experiments/README.md) |
| [03](./notes/03-CIFAR-10与ResNet：从简单CNN到残差网络.md) | CIFAR-10 与 ResNet | 从简单 CNN 到更完整的图像分类工程化 | [CIFAR-10 实验速查](./projects/02-cifar10-cnn-experiments/README.md) |
| [04](./notes/04-自注意力机制：从Q、K、V到缩放点积注意力.md) | 自注意力机制 | 建立 Q、K、V 和缩放点积注意力直觉 | [字符级 Transformer 实验速查](./projects/03-char-transformer-experiments/README.md) |
| [05](./notes/05-Transformer语言模型：从位置编码到最小可训练实现.md) | Transformer 语言模型 | 把注意力落成最小 decoder-only 语言模型 | [字符级 Transformer 实验速查](./projects/03-char-transformer-experiments/README.md) |
| [06](./notes/06-子词级GPT：从BPE到更像真实LLM的训练流程.md) | 子词级 GPT | 补齐 tokenizer、padding 和采样控制 | [子词级 GPT 实验速查](./projects/04-subword-gpt-experiments/README.md) |

## 结果总览

| 主线 | 当前最佳结果 | 主要看点 |
| --- | --- | --- |
| MNIST | `CNN improved` 测试集准确率 `99.47%` | 从 MLP 到 CNN 的第一条完整分类线 |
| CIFAR-10 | `ResNet` 测试集准确率 `95.33%` | 结构升级和训练策略如何共同拉高上限 |
| Character Transformer | `transformer v3` 验证集困惑度 `4.63` | 最小字符级 Transformer 的实现与生成表现 |
| Subword GPT | `subword-gpt v2` 验证集困惑度 `13.19` | 更接近真实 GPT 的 tokenizer 和训练流程 |

语言模型两条结果不能直接横向比较，因为 token 粒度不同。更合适的看法是：

- `char-transformer v3` 更适合看最小语言模型骨架。
- `subword-gpt v2` 更适合看更完整的 GPT 工作流。

## 精选展示

### CIFAR-10 ResNet 预测示例

![CIFAR-10 ResNet predictions](./assets/showcase/cifar10-resnet-predictions.png)

### Character Transformer v3 收敛曲线

![Character Transformer v3 loss curve](./assets/showcase/char-transformer-v3-loss-curve.png)

### Subword GPT v2 收敛曲线

![Subword GPT v2 loss curve](./assets/showcase/subword-gpt-v2-loss-curve.png)

## 快速开始

共享依赖位于 `projects/requirements.txt`。下面两条命令分别对应图像分类线和语言模型线的常用入口。

```bash
cd projects/01-mnist-cnn-experiments
pip install -r ../requirements.txt
python train_cnn.py
```

```bash
cd projects/03-char-transformer-experiments
pip install -r ../requirements.txt
python train_transformer.py
```

运行后，各项目会在自己的目录下生成：

- `data/`：数据集、语料或 tokenizer 文件
- `outputs/<experiment-name>/`：配置、指标、最佳权重、图表和采样结果

这些目录默认仅用于本地运行，不纳入版本控制。

## 仓库结构

```text
DeepLearning-Study-and-Experiments/
├─ assets/
│  ├─ images/        # 笔记配图
│  └─ showcase/      # 精选结果图
├─ notes/            # 主阅读层：原理 + 实验结论 + 运行入口
├─ projects/         # 速查页 + 可运行实验
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 开源协议

本仓库中的代码、笔记与文档内容基于 [MIT License](./LICENSE) 开源。

补充说明：

- 许可证覆盖当前仓库中自行编写和整理的代码、笔记、图示与文档结构。
- 数据集、论文内容、模型思想及其他第三方原始资料，不因本仓库采用 MIT 协议而自动转授额外权利。
