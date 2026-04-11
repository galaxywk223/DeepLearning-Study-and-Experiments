# Transformer 语言模型：从位置编码到最小可训练实现

这篇笔记延续前一篇自注意力笔记，但重点不再只是解释 `Q / K / V`，而是回答另一个更实际的问题：

一个最小可训练的 Transformer 语言模型，到底还需要补上哪些模块？

如果把目标压缩成一句话，就是：

- 输入离散 token，需要先变成可学习向量
- 序列有先后顺序，需要注入位置信息
- 预测下一个 token 时，当前位置不能偷看未来，需要因果掩码
- 单头注意力表达能力有限，需要多头注意力
- 只做注意力还不够，需要前馈网络增强逐位置变换能力
- 深层堆叠要稳定训练，需要残差连接和 LayerNorm

这也是这次实验项目的主线：用字符级语言建模任务，把这些模块真正连起来。

## 从语言模型任务开始

语言模型的目标可以写成：

$$
P(x_1,x_2,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

如果把它放到字符级任务里理解，就是：

- 已知前面的字符
- 预测下一个最可能出现的字符
- 把这个过程不断重复，就能生成连续文本

这件事非常适合拿来解释 Transformer，因为：

- 序列关系明确
- 输出是标准的分类问题
- 因果掩码的作用非常直观

## Token Embedding：先把离散符号变成向量

模型无法直接处理字符 `'a'`、`'b'`、`'?'` 这样的离散符号，所以第一步通常是建立词表，并把每个字符映射到整数 id。

然后通过嵌入层把 id 变成向量：

$$
E=\text{Embedding}(x)
$$

如果批量输入尺寸是：

```text
(batch_size, sequence_length)
```

那么 token embedding 后的张量通常是：

```text
(batch_size, sequence_length, embedding_dim)
```

这里可以把 embedding 理解为离散符号的可学习查表，也是后续注意力和前馈网络真正操作的对象。

## Positional Encoding：告诉模型顺序

自注意力本身只关心“谁和谁相关”，并不天然区分第一个字符和第十个字符。

但语言序列显然有顺序，所以还要加位置信息。常见做法有两种：

### 1. 固定正弦位置编码

经典 Transformer 使用的是正弦和余弦函数：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

优点是：

- 不需要额外学习参数
- 在不同长度上有一定外推性

### 2. 可学习位置编码

另一种更常见于语言模型的方式，是直接为每个位置分配一个可学习向量：

$$
P=\text{Embedding}(position)
$$

然后把 token embedding 和 position embedding 相加：

$$
X=E+P
$$

这次实验代码采用的就是这种方式。原因很简单：实现更直接，也足够清楚地体现“内容信息 + 位置信息”的组合。

## 为什么必须有因果掩码

在语言模型里，位置 `t` 预测的是下一个 token，因此它只能看到：

$$
x_1,x_2,\dots,x_t
$$

不能看到未来的：

$$
x_{t+1},x_{t+2},\dots
$$

因此需要一个下三角掩码：

$$
M=
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 \\
\end{bmatrix}
$$

在计算注意力分数时，把未来位置强行屏蔽：

$$
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V
$$

这一步的本质不是“模型学会不看未来”，而是“结构上禁止它看未来”。

## Multi-Head Attention：不只看一种关联

单头注意力只能在一个投影空间里建模关联，这会让表达能力偏单一。

多头注意力的做法是：

1. 把输入投影成多组 `Q / K / V`
2. 每一组独立计算注意力
3. 把多个头的结果拼接后再线性映射回来

矩阵形式上可以写成：

$$
\text{MultiHead}(X)=\text{Concat}(head_1,\dots,head_h)W_O
$$

其中每个 `head_i` 都是一次独立的 masked self-attention。

## Feed-Forward Network：逐位置的非线性变换

注意力负责的是“位置之间如何交换信息”，但每个位置本身也需要做更强的非线性变换。

所以在每个 Transformer block 里，还会接一个前馈网络：

$$
\text{FFN}(x)=W_2 \sigma(W_1 x + b_1) + b_2
$$

在实现里通常表现为：

- 先把维度升高到 `4 * d_model` 左右
- 经过 `GELU` 或 `ReLU`
- 再投影回原维度

这一层是逐位置独立作用的，不负责位置间通信，但它能显著提高表示能力。

## 残差连接与 LayerNorm：让深层堆叠更稳定

一个标准 Transformer block 常写成：

```text
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

这就是常见的 pre-norm 结构。

残差连接让信息和梯度有一条更直接的通路，LayerNorm 则帮助数值分布保持稳定。对序列模型来说，它比 BatchNorm 更自然，因为它不依赖 batch 统计量。

## 一个最小 Decoder-Only Transformer 长什么样

如果把字符级语言模型的最小结构压缩一下，可以写成：

```text
token ids
-> token embedding
-> position embedding
-> dropout
-> N 个 Transformer block
-> final LayerNorm
-> linear head
-> next-token logits
```

其中每个 block 内部是：

```text
LayerNorm
-> masked multi-head self-attention
-> residual add
-> LayerNorm
-> feed-forward
-> residual add
```

这种结构和编码器-解码器版 Transformer 不同，它只保留了解码器一侧，并且只做因果自注意力。对于语言生成任务，这已经足够构成一个清晰的最小原型。

## 损失函数为什么还是交叉熵

虽然模型比 MLP 或 CNN 复杂很多，但训练目标本质上仍然是分类。

假设词表大小是 `V`，那么每个时间步输出一个长度为 `V` 的 logits 向量。真实标签是“下一个字符”的 id，因此依然使用交叉熵损失：

$$
\mathcal{L} = -\sum_t \log P(x_{t+1}\mid x_{\le t})
$$

实现时通常把：

- logits 从 `(B, T, V)` reshape 成 `(B*T, V)`
- targets 从 `(B, T)` reshape 成 `(B*T)`

然后直接送进 `CrossEntropyLoss`。

## 训练时为什么还要补学习率调度

把结构搭好以后，Transformer 的训练还常常依赖一些工程细节：

- `warmup`
- `weight decay`
- `dropout`
- `gradient clipping`

这次实验里保留了最常见的一版：训练前期线性 warmup，后期用 cosine 衰减学习率。

## 生成文本时发生了什么

训练完成后，生成过程通常是：

1. 给一个起始 prompt
2. 模型输出最后一个位置的 logits
3. 采样得到下一个字符
4. 把新字符拼回序列
5. 重复这个过程

也就是说，训练时模型是“并行看整段，预测整段的下一个字符”；生成时模型是“自回归地一个字符一个字符往后续写”。

温度参数 `temperature` 会直接影响采样风格：

- 较低温度：更保守，更容易重复高概率模式
- 较高温度：更发散，更有随机性，但也更容易失控

## 和前面 CNN 项目的区别

如果和前面的图像分类实验对比，这个字符级 Transformer 项目最值得注意的是三点：

### 1. 输出目标不同

图像分类是整张图输出一个类别；语言模型是序列中每个位置都输出一个分类结果。

### 2. 数据组织不同

这里不是固定样本集，而是从长文本中不断截取长度为 `block_size` 的子序列作为训练片段。

### 3. 评估指标不同

分类任务更常看准确率；语言模型更常看：

- 验证集交叉熵
- perplexity
- 生成文本样例

因此实验产物也会从 `predictions.png` 变成：

- `samples.txt`
- `loss_curve.png`
- `metrics.json`

## 小结

从“理解自注意力”到“写出最小 Transformer”，中间真正需要补齐的是一整条模块链：

- token embedding
- position embedding
- causal mask
- multi-head self-attention
- feed-forward network
- residual + LayerNorm
- 自回归训练与文本生成

如果把这些部分真正实现一遍，你对 Transformer 的理解会从“知道公式”进入“知道它为什么能跑起来，以及训练代码里每一层到底在干什么”。
