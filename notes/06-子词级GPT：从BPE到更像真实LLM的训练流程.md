# 子词级 GPT：从 BPE 到更像真实 LLM 的训练流程

## 本章目标

- 把最小字符级语言模型继续推进到更接近真实 GPT 的训练流程。
- 看懂 tokenizer、special tokens、padding mask 和采样控制为什么会一起出现。
- 建立“模型结构 + 数据管线 + 生成策略”是一条完整工程链路的直觉。

## 本章实验

- 对应项目：[子词级 GPT 实验速查](../experiments/04-subword-gpt-experiments/README.md)
- 本章聚焦：`subword-gpt v1`、`subword-gpt v2`
- 实验产物：tokenizer 文件、指标、最佳权重、生成样例和 loss 曲线

## 关键结果

| 版本 | 词表大小 | 参数量 | 最佳验证困惑度 | 主要看点 |
| --- | ---: | ---: | ---: | --- |
| `subword-gpt v1` | 512 | 6,490,624 | `19.51` | 第一版完整子词级 GPT 基线 |
| `subword-gpt v2` | 512 | 9,194,976 | `13.19` | 在完整工作流下继续提升验证表现 |

<p align="center">
  <img src="../assets/showcase/subword-gpt-v2-loss-curve.png" alt="Subword GPT v2 收敛曲线" width="760" />
</p>

前一篇笔记里，Transformer 语言模型还是字符级的。

那一版很适合解释注意力、位置编码、因果掩码和自回归生成，但它离真实 GPT 还差一段关键距离：

- 词元（token）不是按字符切，而是更常见地按子词切
- 序列里通常会有特殊词元（special tokens）
- 批次（batch）不一定都是等长，需要填充（padding）
- 损失（loss）不能把 pad 位置也算进去
- 生成时也不只是调一个温度参数（temperature）

所以这一步的重点，不再只是“Transformer block 怎么堆”，而是：

把一个最小语言模型，推进到更接近真实工程形态的分词器（tokenizer）+ GPT 流程。

## 为什么要从字符级走向子词级

字符级模型最大的优点是直观：

- 词表很小
- 不会有 OOV
- 每个训练样本的编码过程非常简单

但它的问题也很明显：

- 序列太长
- 一个单词会被拆成很多字符
- 模型需要花大量容量去学习拼写层面的局部规律

例如单词 `king`，字符级要拆成：

```text
k i n g
```

而子词级分词器（tokenizer）可能会把它保留成：

```text
king
```

或者拆成更有统计意义的片段：

```text
ki ng
```

这样一来，模型能用更短的序列看到更长的语义范围。

## BPE 在做什么

字节对编码（Byte Pair Encoding，BPE）的直觉可以概括成一句话：

从最小单位开始，反复把最常一起出现的相邻词元（token）合并成一个新词元。

如果起点是 byte 字节，那么初始词表就是：

```text
0 ~ 255
```

也就是所有可能的 UTF-8 byte。

随后不断统计相邻词元对：

```text
(x_i, x_{i+1})
```

每轮把频率最高的一对合并：

```text
(a, b) -> c
```

随着 merge 合并规则越来越多，常见片段会逐渐变成更长的词元。

## 为什么这里使用字节级 BPE（byte-level BPE）

和传统词级词表相比，字节级 BPE（byte-level BPE）有两个很实际的优点：

### 1. 不会有真正的 OOV

因为任何文本都能先表示成 UTF-8 byte 序列，所以最坏情况也能退回到 byte 级编码。

### 2. 实现路径更统一

这一方案不需要先决定“按空格切词”还是“按字符切词”，而是直接从 byte 出发，再让 merge 规则自己长出来。

这也是很多现代分词器（tokenizer）采用 byte 作为底层单位的原因之一。

## 特殊词元（special tokens）为什么重要

从字符级最小实验走向更真实的 GPT，通常都会补上几个特殊词元：

- `<pad>`：把不同长度样本 pad 到同样长度
- `<bos>`：表示序列开始
- `<eos>`：表示序列结束

这些词元的作用并不神秘，但非常关键。

### `<pad>`

如果一个批次（batch）里的样本长度不同，例如：

```text
[12, 51, 88, 20]
[12, 51]
```

就需要填充（pad）成统一形状：

```text
[12, 51, 88, 20]
[12, 51, <pad>, <pad>]
```

### `<bos>` 和 `<eos>`

这两个词元让模型更清楚序列边界在哪里，也方便生成阶段从一个明确起点开始，或者在看到结束标记后提前停止。

## 填充（padding）为什么不能只“补齐”了事

语言模型的初版实现常见一个问题：样本 pad 到同样长度后，后续两件事容易被忽略：

### 1. 注意力机制（Attention）不能看见 pad

如果 pad 位置也参与注意力，模型会把无意义占位符当成上下文的一部分。

所以在注意力计算里，要把：

- 因果掩码（causal mask）
- 填充掩码（padding mask）

一起用起来。

也就是说，某个位置只能看：

- 它前面的真实词元
- 不能看未来
- 也不能看 pad

### 2. 损失（loss）不能计算 pad

假设目标序列里有一部分只是为了补齐长度：

```text
[51, 88, 20, <pad>, <pad>]
```

如果这些 `<pad>` 位置也一并送进交叉熵，模型就会被迫学习“如何预测 pad”，这会污染训练目标。

因此更合理的做法是：

- 对 pad 位置使用 `ignore_index`
- 只在真实词元上计算损失

## 一个更像真实 GPT 的最小结构

把子词级 GPT 的训练流程压缩一下，可以写成：

```text
raw text
-> byte-level BPE tokenizer
-> token ids + special tokens
-> variable-length windows
-> padding to block_size
-> token embedding + position embedding
-> decoder-only Transformer blocks
-> LayerNorm
-> lm head
-> next-token cross entropy
```

和字符级实验相比，这里新增的不是注意力公式，而是数据和训练流程的“现实细节”。

## 权重绑定（Weight Tying）：为什么常把输出头和嵌入层绑在一起

在语言模型里，输入嵌入（embedding）和输出分类头其实都在和“词表”打交道。

因此常见做法是：

- 输入侧：`Embedding(vocab_size, d_model)`
- 输出侧：`Linear(d_model, vocab_size)`

并让两者共享同一份权重。

这叫权重绑定（weight tying）。

它的直觉是：

- 输入时，词元要被映射进语义空间
- 输出时，隐藏状态要再投影回词表空间

这两个空间本来就高度相关，共享参数通常更合理，也能减少参数量。

## 生成时为什么要有 top-k 和 top-p 采样

只用 temperature 当然可以采样，但它控制的只是概率分布整体的“平滑程度”。

真实一点的生成通常还会加：

### top-k 采样

只保留概率最高的前 `k` 个词元，再从里面采样。

这会强行截掉长尾噪声，输出通常更稳。

### top-p 采样

按概率从大到小累加，只保留累计概率达到阈值 `p` 的那部分词元。

这是一种动态截断方式：

- 分布很尖锐时，保留候选会比较少
- 分布很平时，保留候选会更多

所以 `top-p` 往往比固定 `top-k` 更灵活。

## 和前一个字符级项目相比，真正升级了什么

把这一步视为“从教学版 Transformer 到更像真实 GPT 的过渡”时，核心升级主要有五个：

### 1. 词元单位升级

从字符变成子词，序列更短，语义密度更高。

### 2. 数据管线升级

不再只是“长文本随机截一段”，而是引入：

- 分词器（tokenizer）
- 特殊词元（special tokens）
- 填充（padding）
- 注意力掩码（attention mask）

### 3. 损失计算更规范

pad 位置不再参与损失计算。

### 4. 生成控制更真实

不只看温度参数（temperature），还能加 `top-k / top-p`。

### 5. 模型实现更接近 GPT 常见做法

例如权重绑定（weight tying）、能识别填充位置的注意力（padding-aware attention）等细节，都比最小字符模型更接近真实项目。

## 小结

字符级 Transformer 非常适合作为入门第一站，因为它把注意力机制解释得足够清楚。

但继续往前推进时，下一步最值得补的并不是再横向学一个完全不同的模型，而是把语言模型这条线做得更完整：

- 学会分词器（tokenizer）
- 学会特殊词元（special tokens）
- 学会能识别填充位置的训练（padding-aware training）
- 学会更合理的采样（sampling）
- 学会把模型代码和数据代码衔接成一条更像真实 GPT 的训练路径

完成这一步之后，对 GPT 的理解会更接近“它在工程上到底是怎么被训练出来的”，而不只是停留在注意力公式层面。

## 代码入口

- `experiments/04-subword-gpt-experiments/train_gpt.py`：训练入口
- `experiments/04-subword-gpt-experiments/generate_samples.py`：生成样例导出入口
- `experiments/04-subword-gpt-experiments/subword_gpt_experiments/tokenizer.py`：BPE tokenizer 逻辑
- `experiments/04-subword-gpt-experiments/subword_gpt_experiments/models.py`：模型定义
- `experiments/04-subword-gpt-experiments/subword_gpt_experiments/runner.py`：训练主流程

## 继续阅读

- 上一章：[05-Transformer语言模型：从位置编码到最小可训练实现](./05-Transformer语言模型：从位置编码到最小可训练实现.md)
- 项目速查：[子词级 GPT 实验速查](../experiments/04-subword-gpt-experiments/README.md)

## 如何运行

```bash
cd experiments/04-subword-gpt-experiments
pip install -r ../requirements.txt
python train_gpt.py
```

不同温度生成结果的导出命令如下：

```bash
python generate_samples.py --run-dir outputs/<experiment-name> --temperatures 0.6 0.8 1.0 --top-k 40 --top-p 0.95
```
