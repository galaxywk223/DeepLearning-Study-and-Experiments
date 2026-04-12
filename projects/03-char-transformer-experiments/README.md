# 字符级 Transformer 实验

这个项目实现的是一条最小字符级语言模型实验线，从 bigram 到 Transformer。

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

- 从 `bigram` 到 `transformer v3`，验证集困惑度从 `12.73` 下降到 `4.63`。
- `transformer v3` 已经能比较稳定地学到 Shakespeare 风格对白的排版和局部句式。
- 这组结果可以和对应笔记一起看。

## 精选展示

![Character Transformer v3 loss curve](../../assets/showcase/char-transformer-v3-loss-curve.png)

这张图展示了 `transformer v3` 的收敛情况。

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

这段样例还谈不上稳定生成，但已经能看出：

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
