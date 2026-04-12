# 深度学习学习笔记与实验

这个仓库整理了我在深度学习学习过程中的几组实验和配套笔记。

内容分成三层：

- 首页：先看项目概况、代表结果和入口。
- 项目页：看每个实验的结果、图表和运行方式。
- 笔记页：看原理推导和实现细节。

## 内容结构

| 路线阶段 | 代表内容 | 你会看到什么 | 入口 |
| --- | --- | --- | --- |
| 1. MLP 与 CNN 入门 | MNIST | 从全连接基线到卷积网络，建立最小训练流程 | [MNIST 项目](./projects/01-mnist-cnn-experiments/README.md) |
| 2. 图像分类工程化 | CIFAR-10 | 从简单 CNN 到改进版再到 ResNet，展示结构升级和训练策略带来的提升 | [CIFAR-10 项目](./projects/02-cifar10-cnn-experiments/README.md) |
| 3. 最小 Transformer 语言模型 | Character Transformer | 从 bigram baseline 到 decoder-only Transformer，观察困惑度与生成质量变化 | [字符级 Transformer 项目](./projects/03-char-transformer-experiments/README.md) |
| 4. 更接近真实 GPT 的流程 | Subword GPT | 加入 BPE tokenizer、special tokens、padding mask 和采样控制 | [子词级 GPT 项目](./projects/04-subword-gpt-experiments/README.md) |

## 代表结果

| 主线 | 最佳结果 | 主要内容 |
| --- | --- | --- |
| MNIST | `CNN improved` 测试集准确率 `99.47%` | 从 MLP 到 CNN 的一条基础实验线 |
| CIFAR-10 | `ResNet` 测试集准确率 `95.33%` | 从简单 CNN 到 ResNet 的性能迭代 |
| Character Transformer | `transformer v3` 验证集困惑度 `4.63` | 最小字符级 Transformer 实现与生成结果 |
| Subword GPT | `subword-gpt v2` 验证集困惑度 `13.19` | 子词级 tokenizer 和 GPT 训练流程 |

## 语言模型速览

| 模型 | token 粒度 | 参数量 | 最佳验证困惑度 | 主要特点 |
| --- | --- | ---: | ---: | --- |
| `char-transformer v3` | 字符级 | 2,286,593 | `4.63` | 生成观感更成熟，结构更直观 |
| `subword-gpt v2` | 子词级 BPE | 9,194,976 | `13.19` | tokenizer、padding mask 与采样控制更完整 |

这两个数字不能直接比较，因为 token 粒度不同。更合适的看法是：

- `char-transformer v3` 更偏最小语言模型本身。
- `subword-gpt v2` 更偏完整训练流程。

## 精选展示

### CIFAR-10 ResNet 预测示例

![CIFAR-10 ResNet predictions](./assets/showcase/cifar10-resnet-predictions.png)

### Character Transformer v3 收敛曲线

![Character Transformer v3 loss curve](./assets/showcase/char-transformer-v3-loss-curve.png)

### Subword GPT v2 收敛曲线

![Subword GPT v2 loss curve](./assets/showcase/subword-gpt-v2-loss-curve.png)

## 推荐阅读顺序

第一次看这个仓库，可以按下面顺序读：

1. 先读 [学习路线与项目导航](./docs/学习路线与项目导航.md)，快速建立全局印象。
2. 再看 [实验结果总览](./docs/实验结果总览.md)，直接比较四条主线的结果和展示重点。
3. 对某个方向感兴趣后，进入对应项目页看实现与运行方式。
4. 如果想看原理推导和学习过程，再进入 [笔记索引](./notes/README.md)。

## 快速开始

共享依赖文件位于 `projects/requirements.txt`。

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

说明：

- 每个项目默认把数据和输出写入各自项目目录下的 `data/` 与 `outputs/`。
- 这些目录默认是本地运行目录，不纳入版本控制。
- 仓库只保留 `assets/showcase/` 下的少量结果图。

## 仓库结构

```text
DeepLearning-Study-and-Experiments/
├─ assets/
│  ├─ images/        # 笔记配图
│  └─ showcase/      # GitHub 展示用精选结果
├─ docs/             # 导航与结果总览
├─ notes/            # 详细学习笔记
├─ projects/         # 四个可运行实验项目
├─ .gitignore
├─ LICENSE
└─ README.md
```

## 开源协议

本仓库中的代码、笔记与文档内容基于 [MIT License](./LICENSE) 开源。

补充说明：

- 本仓库的许可证覆盖当前仓库中自行编写和整理的代码、笔记、图示与文档结构。
- 数据集、论文内容、模型思想及其他第三方原始资料，不因本仓库采用 MIT 协议而自动转授额外权利。
