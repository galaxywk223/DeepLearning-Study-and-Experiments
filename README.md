# 深度学习学习笔记与实验

这是一个面向 GitHub 展示的深度学习学习仓库。

它不只放代码，也把“从基础图像分类到最小语言模型，再到更接近真实 GPT 工作流”的学习路线整理成了可阅读、可运行、可对比的三层结构：

- 首页：先快速看懂我做了哪些项目、结果到什么程度、推荐从哪里开始看。
- 项目页：看每个实验的定位、关键结果、代表图表和运行方式。
- 笔记页：看完整原理推导、实现思路和学习过程。

## 这个仓库有什么

| 路线阶段 | 代表内容 | 你会看到什么 | 入口 |
| --- | --- | --- | --- |
| 1. MLP 与 CNN 入门 | MNIST | 从全连接基线到卷积网络，建立最小训练流程 | [MNIST 项目](./projects/01-mnist-cnn-experiments/README.md) |
| 2. 图像分类工程化 | CIFAR-10 | 从简单 CNN 到改进版再到 ResNet，展示结构升级和训练策略带来的提升 | [CIFAR-10 项目](./projects/02-cifar10-cnn-experiments/README.md) |
| 3. 最小 Transformer 语言模型 | Character Transformer | 从 bigram baseline 到 decoder-only Transformer，观察困惑度与生成质量变化 | [字符级 Transformer 项目](./projects/03-char-transformer-experiments/README.md) |
| 4. 更接近真实 GPT 的流程 | Subword GPT | 加入 BPE tokenizer、special tokens、padding mask 和采样控制 | [子词级 GPT 项目](./projects/04-subword-gpt-experiments/README.md) |

## 代表结果

| 主线 | 最佳结果 | 想表达的核心点 |
| --- | --- | --- |
| MNIST | `CNN improved` 测试集准确率 `99.47%` | 从最小 MLP 到卷积网络，训练流程和可视化链路完整跑通 |
| CIFAR-10 | `ResNet` 测试集准确率 `95.33%` | 清楚展示“基线有限，工程优化和结构升级有效” |
| Character Transformer | `transformer v3` 验证集困惑度 `4.63` | 最小字符级 Transformer 已能学到对白格式与局部句式 |
| Subword GPT | `subword-gpt v1` 验证集困惑度 `19.51` | 工程形态明显更接近真实 GPT 训练工作流 |

## 精选展示

### CIFAR-10 ResNet 预测示例

![CIFAR-10 ResNet predictions](./assets/showcase/cifar10-resnet-predictions.png)

### Character Transformer v3 收敛曲线

![Character Transformer v3 loss curve](./assets/showcase/char-transformer-v3-loss-curve.png)

## 推荐阅读顺序

如果你想快速了解这个仓库，建议按下面顺序看：

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
- 仓库只保留 `assets/showcase/` 下的精选展示资源，避免把原始数据、完整权重和训练残留混进公开目录。

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
