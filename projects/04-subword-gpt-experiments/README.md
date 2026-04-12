# 子词级 GPT 实验

这个项目是在字符级 Transformer 之后继续往前走的一步，补上了 tokenizer、padding 和采样控制这些部件。

## 项目定位

- 使用 `byte-level BPE` 从原始文本训练 tokenizer。
- 显式引入 `<pad> / <bos> / <eos>`。
- 训练 `decoder-only GPT`，保留因果掩码。
- 在训练与评估阶段处理 `padding mask`。
- 在生成阶段支持 `temperature + top-k + top-p`。

重点不只是在指标上，也在于训练和生成流程更完整。

## 核心结果

| 运行名 | 设备 | 词表大小 | 参数量 | 最佳验证损失 | 最佳验证困惑度 | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `tinyshakespeare-subword-gpt-v1` | CUDA | 512 | 6,490,624 | `2.9711` | `19.51` | 第一版正式结果，证明子词级 GPT 训练链路已经完整跑通 |
| `tinyshakespeare-subword-gpt-v2` | CUDA | 512 | 9,194,976 | `2.5797` | `13.19` | 第二版正式结果，最佳点出现在第 18 轮，验证表现明显优于 `v1` |
| `smoke-subword-gpt` | CPU | 300 | 121,344 | `5.6604` | `287.27` | 用于链路冒烟验证 |

- tokenizer 从字符级切到了子词级
- 训练流程开始处理 variable-length batch 与 padding
- 生成控制开始接近真实 LLM 常见工作流

## 精选展示

![Subword GPT v2 loss curve](../../assets/showcase/subword-gpt-v2-loss-curve.png)

这张曲线对应当前保留在仓库里的 `v2` 结果。

## 阶段对比

| 版本 | 参数量 | 最佳验证损失 | 最佳验证困惑度 | 相比上一版的变化 |
| --- | ---: | ---: | ---: | --- |
| `subword-gpt v1` | 6,490,624 | `2.9711` | `19.51` | 建立第一版可公开展示的子词级 GPT 基线 |
| `subword-gpt v2` | 9,194,976 | `2.5797` | `13.19` | 模型容量和训练强度上去后，验证困惑度进一步下降 `6.32` |

如果把它和字符级 `char-transformer v3` 放在一起看：

| 模型 | token 粒度 | 参数量 | 最佳验证困惑度 | 主要区别 |
| --- | --- | ---: | ---: | --- |
| `char-transformer v3` | 字符级 | 2,286,593 | `4.63` | 生成观感更成熟 |
| `subword-gpt v2` | 子词级 BPE | 9,194,976 | `13.19` | 训练与生成流程更完整 |

这里不建议把两个困惑度直接当成绝对高低，因为 token 粒度不同。更合适的是把生成结果和训练流程分开看。

## 生成样例摘录

下面这段摘自 `subword-gpt v2` 在 `temperature = 0.8` 时的正式采样结果：

```text
ROMEO:
Peace, for no good time: for you are bear not
false the pleasure from my tongue.
```

这说明 `v2` 已经能稳定产出更像对白开头的短句，但语义连贯性仍然有限。

和字符级 `transformer v3` 相比：

- 字符级模型在当前设置下生成质量更成熟
- 子词级模型更接近真实 GPT 的训练流程
- `v2` 相比 `v1` 的验证结果也有明显进步

## 如何运行

```bash
pip install -r ../requirements.txt
python train_gpt.py
```

如果想导出不同温度下的生成结果：

```bash
python generate_samples.py --run-dir outputs/<experiment-name> --temperatures 0.6 0.8 1.0 --top-k 40 --top-p 0.95
```

运行后会在本项目目录下自动生成：

- `data/`：语料与 tokenizer 文件
- `outputs/<experiment-name>/`：配置、指标、最佳权重、生成样例和 loss 曲线

这些目录默认仅用于本地运行，不纳入版本控制。

## 代码结构

```text
04-subword-gpt-experiments/
├─ subword_gpt_experiments/
│  ├─ cli.py
│  ├─ config.py
│  ├─ data.py
│  ├─ engine.py
│  ├─ generate.py
│  ├─ models.py
│  ├─ runner.py
│  ├─ tokenizer.py
│  ├─ utils.py
│  └─ visualize.py
├─ generate_samples.py
└─ train_gpt.py
```

## 延伸阅读

- 理论与实现说明见：[05-子词级GPT：从BPE到更像真实LLM的训练流程](../../notes/05-子词级GPT：从BPE到更像真实LLM的训练流程.md)
- 和字符级模型的并排比较见：[实验结果总览](../../docs/实验结果总览.md)
- 如果想先理解最小 Transformer 骨架，建议先看：[字符级 Transformer 实验](../03-char-transformer-experiments/README.md)
