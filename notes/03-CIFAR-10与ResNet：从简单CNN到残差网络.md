# CIFAR-10 与 ResNet：从简单 CNN 到残差网络

## 本章目标

- 从 MNIST 的入门分类任务继续往前走，理解为什么 CIFAR-10 会明显更难。
- 看懂简单 CNN、改进版 CNN 和 ResNet 分别解决什么问题。
- 把“模型结构升级”和“训练策略升级”放在同一条实验线上观察。

## 本章实验

- 对应项目：[CIFAR-10 实验速查](../experiments/02-cifar10-cnn-experiments/README.md)
- 本章聚焦：`baseline`、`improved`、`resnet`
- 你会产出：配置、指标、最佳权重和预测可视化

## 关键结果

| 版本 | 最佳测试准确率 | 关键变化 |
| --- | ---: | --- |
| `baseline` | `73.25%` | 简单 CNN，无增强 |
| `improved` | `87.35%` | 更深 CNN + BatchNorm + Dropout + AdamW + Cosine + AMP |
| `resnet` | `95.33%` | ResNet + SGD + MultiStep + AMP + RandomErasing |

<p align="center">
  <img src="../assets/showcase/cifar10-resnet-predictions.png" alt="CIFAR-10 ResNet 预测结果" width="920" />
</p>

## 为什么 CIFAR-10 比 MNIST 难得多

MNIST 的输入是单通道灰度图，背景简单，类别之间差异也相对明显。CIFAR-10 则不同：

- 输入变成 `32x32x3` 彩色图像，信息密度更高
- 类别之间的形状差异没有手写数字那么直接
- 同一类别内部变化更大，例如猫、狗、汽车在姿态和背景上都可能差很多

这意味着模型不能只依赖几个局部模板，而要更稳定地抽取层级特征。

## 从简单 CNN 到改进版 CNN，真正补了什么

`baseline` 的意义不是追求极限，而是留下一个最清晰的对照组。到了 `improved` 版本，主要补的是两类东西：

### 1. 模型结构上的补强

- 网络更深，表示能力更强
- 引入 `BatchNorm`，让训练更稳定
- 引入 `Dropout`，降低过拟合风险

### 2. 训练流程上的补强

- 数据增强不再缺席
- 优化器切到 `AdamW`
- 学习率调度改成 `CosineAnnealing`
- 训练里加入 `AMP`，提高吞吐

这一步很适合用来建立一个直觉：很多时候效果提升不是只来自“换了一个新模型名”，而是结构和训练流程一起变得更完整了。

## 为什么还要引入 ResNet

当网络继续加深时，普通 CNN 容易出现优化困难。ResNet 的核心直觉是让一部分信息走一条更短的恒等路径：

$$
y = F(x) + x
$$

这里的 `F(x)` 是残差分支学到的增量，`x` 是直接跳连过来的输入。这样做的好处是：

- 梯度更容易往前传
- 深层网络更容易训练
- 网络可以在“保留原信息”和“学习修正量”之间取得平衡

在 CIFAR-10 这条实验线上，`resnet` 把测试准确率进一步推进到 `95.33%`，说明当任务复杂度上来以后，结构升级本身会成为更强的决定因素。

## 这条实验线最值得带走的结论

如果把三版实验放在一起看，可以读出三层信息：

1. `baseline` 说明最小 CNN 也能工作，但上限有限。
2. `improved` 说明训练策略和常见工程细节会带来非常实在的收益。
3. `resnet` 说明当任务更复杂时，模型骨架本身会显著影响最终上限。

也就是说，这一章不只是“把准确率做高了一点”，而是在图像分类这条线上补出了从入门到工程化之间的过渡层。

## 代码入口

- `experiments/02-cifar10-cnn-experiments/train_baseline.py`：简单 CNN 训练入口
- `experiments/02-cifar10-cnn-experiments/train_improved.py`：改进版 CNN 训练入口
- `experiments/02-cifar10-cnn-experiments/train_resnet.py`：ResNet 训练入口
- `experiments/02-cifar10-cnn-experiments/cifar10_experiments/models.py`：模型定义
- `experiments/02-cifar10-cnn-experiments/cifar10_experiments/runner.py`：训练主流程

## 继续阅读

- 上一章：[02-CNN与MNIST：从卷积直觉到图像分类升级](./02-CNN与MNIST：从卷积直觉到图像分类升级.md)
- 下一章：[04-自注意力机制：从Q、K、V到缩放点积注意力](./04-自注意力机制：从Q、K、V到缩放点积注意力.md)
- 项目速查：[CIFAR-10 实验速查](../experiments/02-cifar10-cnn-experiments/README.md)

## 如何运行

```bash
cd experiments/02-cifar10-cnn-experiments
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```
