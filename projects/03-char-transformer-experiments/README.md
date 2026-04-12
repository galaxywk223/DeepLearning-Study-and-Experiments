# 字符级 Transformer 实验

这个项目把“自注意力公式”真正推进成一个可训练、可生成、可对比的最小语言模型实现。

它的展示重点不是做一个大模型，而是把 Transformer 语言模型最核心的一条实现主线彻底走通。

## 项目定位

- `bigram`：最小字符级 next-token baseline，用来建立语言模型的起点。
- `transformer`：加入位置编码、因果掩码、多头注意力、FFN 和 LayerNorm。
- `transformer v2 / v3`：在同一条线上继续扩大上下文和容量，观察困惑度与生成质量怎么变化。

## 核心结果

| 版本 | 轮数 | 批大小 | `block_size` | 参数量 | 最佳验证损失 | 最佳验证困惑度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bigram` | 4 | 64 | 64 | 4,225 | `2.5441` | `12.73` |
| `transformer` | 8 | 64 | 128 | 826,433 | `2.1182` | `8.32` |
| `transformer v2` | 12 | 48 | 192 | 2,286,593 | `1.6733` | `5.33` |
| `transformer v3` | 18 | 48 | 192 | 2,286,593 | `1.5333` | `4.63` |

这条线最重要的信息是：

- 从 `bigram` 到 `transformer v3`，验证集困惑度从 `12.73` 下降到 `4.63`。
- `transformer v3` 已经能比较稳定地学到 Shakespeare 风格对白的排版和局部句式。
- 这个项目非常适合和笔记一起阅读，因为“公式”和“代码”可以一一对上。

## 精选展示

![Character Transformer v3 loss curve](../../assets/showcase/char-transformer-v3-loss-curve.png)

上面这张曲线图来自整理后的精选展示资源，用来保留对外最有价值的收敛证据。

## 生成样例摘录

下面这段摘自 `transformer v3` 在 `temperature = 0.75` 时的生成结果：

```text
ROMEO:
See, she hath princely, proper, I have desperous
The lives man of Rome, our voices them ne'er brook
Not be the fight to the blood of this breads:
My good daughter them against with soft a noble offence,
As I am, the should seem down, as which may me?
```

它当然还不是真正可用的语言模型，但已经能明显看出：

- 对白格式开始稳定
- 角色说话的局部语气更像戏剧文本
- 相比 bigram，长距离结构感更强

## 如何运行

```bash
pip install -r ../requirements.txt
python train_bigram.py
python train_transformer.py
```

训练完成后可以单独导出不同温度下的生成样例：

```bash
python generate_samples.py --run-dir outputs/<experiment-name> --temperatures 0.6 0.75 0.9
```

运行后会在本项目目录下自动生成：

- `data/`：语料文件
- `outputs/<experiment-name>/`：配置、指标、最佳权重、生成样例和 loss 曲线

这些目录默认只用于本地运行，不纳入版本控制。

## 代码结构

```text
03-char-transformer-experiments/
├─ char_transformer_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ generate.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ utils.py
│  └─ visualize.py
├─ generate_samples.py
├─ train_bigram.py
└─ train_transformer.py
```

## 延伸阅读

- 自注意力原理见：[03-自注意力机制：从Q、K、V到缩放点积注意力](../../notes/03-自注意力机制：从Q、K、V到缩放点积注意力.md)
- 完整实现讲解见：[04-Transformer语言模型：从位置编码到最小可训练实现](../../notes/04-Transformer语言模型：从位置编码到最小可训练实现.md)
- 和子词级模型的对照见：[实验结果总览](../../docs/实验结果总览.md)
