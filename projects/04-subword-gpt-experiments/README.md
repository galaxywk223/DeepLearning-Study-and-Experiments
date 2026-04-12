# 子词级 GPT 实验（Subword GPT）

这个项目把前一个字符级 Transformer 继续推进到更接近真实 LLM 训练流程的一步：

- 使用 `byte-level BPE` 从原始文本自动训练 tokenizer
- 引入 `special tokens`：`<pad> / <bos> / <eos>`
- 训练 `decoder-only GPT`，保留因果掩码
- 在训练和评估阶段显式处理 `padding mask`
- 在生成阶段支持 `temperature + top-k + top-p` 采样

相比前一个字符级项目，这一版不再只回答“最小 Transformer 怎么写”，而是进一步回答：

一个更像真实 GPT 工作流的小型项目，到底还需要补哪些部件？

## 这个项目补上的关键能力

- `Tokenizer`：不再逐字符建模，而是从 UTF-8 byte 序列学习 BPE merge
- `Special tokens`：训练和生成都显式区分 padding、序列起点和终点
- `Variable-length batches`：随机采样不同长度窗口，再 pad 到统一 `block_size`
- `Padding-aware loss`：交叉熵忽略 `<pad>` 位置
- `Weight tying`：输出头与 token embedding 共享权重
- `Sampling controls`：支持 `top-k` 和 `top-p`

## 默认实验定位

- 语料：`tiny Shakespeare`
- tokenizer：`byte-level BPE`
- 目标词表大小：`512`（包含 special tokens）
- 模型：`6-layer decoder-only GPT`
- 位置编码：`learned positional embedding`
- 优化器：`AdamW`
- 学习率策略：`warmup + cosine decay`
- 正则化：`dropout + weight decay + gradient clipping`

这个默认配置仍然偏教学和可复现优先，不是追求大规模生成质量。

## 当前结果

以下结果来自 `2026-04-12` 本地环境实际运行产物。

| 运行名                    | 设备 | 词表大小 | 参数量    | 最佳验证损失 | 最佳验证困惑度 | 说明                                                 |
| ------------------------- | ---- | -------: | --------: | -----------: | -------------: | ---------------------------------------------------- |
| tinyshakespeare-subword-gpt-v1 | CUDA |      512 | 6,490,624 |       2.9711 |          19.51 | `12 epochs / 250 steps`，正式训练结果，最佳点在第 `11` 轮 |
| smoke-subword-gpt         | CPU  |      300 |   121,344 |       5.6604 |         287.27 | `1 epoch / 2 steps`，用于链路冒烟验证                |

## 与字符级 Transformer 的并排对比

为了更清楚地展示这一步升级到底带来了什么，可以把 [03-char-transformer-experiments/README.md](../03-char-transformer-experiments/README.md) 里的 `transformer v3` 和这次的 `subword-gpt v1` 放在一起看。

需要先说明一点：

- 二者的 token 粒度不同
- 因此 `perplexity` 不能简单看成“谁更小谁就绝对更好”
- 更重要的是比较它们分别解决了什么问题

| 模型           | 词元粒度（Tokenization） | 参数量    | `block_size` | 最佳验证损失 | 最佳验证困惑度 | 主要收益                                                                   |
| -------------- | ------------------------ | --------: | -----------: | -----------: | -------------: | -------------------------------------------------------------------------- |
| transformer v3 | character-level          | 2,286,593 |          192 |       1.5333 |           4.63 | 把最小字符级 Transformer 训练到较强生成质量，验证了自注意力主线可以真正跑通 |
| subword-gpt v1 | byte-level BPE           | 6,490,624 |          160 |       2.9711 |          19.51 | 把项目推进到更像真实 GPT 的工作流，补齐 tokenizer、special tokens 和 padding |

这个对比更准确的结论是：

- `transformer v3` 在当前 tiny Shakespeare 设置下，字符级生成质量更成熟
- `subword-gpt v1` 的核心价值不只是指标，而是工程范式升级
- 后者为继续扩展到更大语料、更长上下文和更规范实验打下了更好的骨架

当前仓库主要提供：

- 一套完整可运行的 tokenizer + GPT 代码
- 自动落盘的 `config / tokenizer / metrics / best_model / samples / loss_curve`
- 单独的采样脚本，用于对比不同 `temperature`

这版 `v1` 的训练曲线比较健康：

- 验证集 loss 从第 `1` 轮的 `4.0338` 下降到第 `11` 轮的 `2.9711`
- 最后一轮略回升到 `2.9847`，说明最佳 checkpoint 已经出现在训练后半段
- 最终 `perplexity` 从早期的 `56.48` 降到 `19.51`

从生成样例看，这个模型已经不再是纯乱码，而是进入了“短句可读但仍不稳定”的阶段。例如：

```text
ROMEO:
Should been not so'er, but sweet they well.

ROMEO:
Well, I'll should been, 'tis great for hand?
```

这说明模型已经学到了 Shakespeare 对白的局部句式、标点和角色台词格式，但离更稳定的长句连贯性还有明显距离。

## 运行

```bash
cd projects/04-subword-gpt-experiments
pip install -r ../requirements.txt
python train_gpt.py --experiment-name tinyshakespeare-subword-gpt-v1 --epochs 12 --steps-per-epoch 250 --eval-steps 50 --batch-size 12 --grad-accum-steps 2 --block-size 160 --min-sequence-length 48 --embedding-dim 256 --num-heads 8 --num-layers 8 --dropout 0.15 --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 150 --weight-decay 0.1 --grad-clip 1.0 --tokenizer-vocab-size 512 --device cuda --use-amp --max-new-tokens 240 --temperature 0.8 --top-k 40 --top-p 0.95
```

轻量开发配置：

```bash
python train_gpt.py ^
  --experiment-name tinyshakespeare-subword-gpt-dev ^
  --epochs 2 ^
  --steps-per-epoch 20 ^
  --eval-steps 5 ^
  --batch-size 8 ^
  --block-size 96 ^
  --min-sequence-length 24 ^
  --embedding-dim 96 ^
  --num-heads 4 ^
  --num-layers 3 ^
  --tokenizer-vocab-size 384 ^
  --device cpu ^
  --no-use-amp
```

训练完成后，可以批量导出不同采样温度下的文本：

```bash
python generate_samples.py ^
  --run-dir outputs/tinyshakespeare-subword-gpt-v1 ^
  --temperatures 0.6 0.8 1.0 ^
  --top-k 40 ^
  --top-p 0.95
```

说明：

- 如果 `data/` 下不存在语料文件，会自动下载 `tiny Shakespeare`
- 如果 `data/` 下不存在 tokenizer 文件，会先训练 BPE tokenizer 并保存为 JSON
- 默认数据和输出都落在当前项目目录，不会污染仓库根目录

## 输出文件

每次运行都会写入 `outputs/<experiment-name>/`：

- `config.json`：本次实验配置
- `metrics.json`：训练历史、最佳验证损失、最终 perplexity
- `best_model.pt`：最佳 checkpoint
- `samples.txt`：给定 prompt 的生成样例
- `loss_curve.png`：训练集 / 验证集 loss 曲线
- `temperature_sweep.txt`：使用 `generate_samples.py` 导出的多温度文本对比

其中 tokenizer 会写入 `data/<tokenizer-name>.json`，包含：

- learned token byte 序列
- merge 规则顺序
- special tokens

## 项目结构

```text
04-subword-gpt-experiments/
├─ data/                                 # local, gitignored
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
├─ outputs/                              # local, gitignored
├─ generate_samples.py
├─ train_gpt.py
└─ ../requirements.txt
```

## 源码职责

- `train_gpt.py`：训练入口
- `generate_samples.py`：读取已训练模型，导出多温度文本样例
- `subword_gpt_experiments/tokenizer.py`：byte-level BPE tokenizer 训练、保存与加载
- `subword_gpt_experiments/data.py`：语料下载、文档切分、tokenize、随机 batch 采样
- `subword_gpt_experiments/models.py`：decoder-only GPT、因果掩码、sampling 策略
- `subword_gpt_experiments/engine.py`：训练循环、评估和学习率调度
- `subword_gpt_experiments/runner.py`：实验编排、checkpoint、metrics 和样例输出
- `subword_gpt_experiments/visualize.py`：loss 曲线保存

## 默认配置

`v1` 正式结果对应配置

- tokenizer vocab：`512`
- batch size：`12`
- grad accumulation：`2`
- epochs：`12`
- steps per epoch：`250`
- `block_size`：`160`
- `embedding_dim`：`256`
- `num_heads`：`8`
- `num_layers`：`8`
- `dropout`：`0.15`
- sampling：`temperature=0.8, top_k=40, top_p=0.95`

## 工程说明

- tokenizer 使用 UTF-8 byte 作为基础词表，因此不会出现传统词级 tokenizer 的 OOV 问题
- 训练 batch 不是固定长度纯裁剪，而是先随机取不同长度窗口，再统一 pad 到 `block_size`
- self-attention 同时应用 `causal mask` 和 `padding mask`
- loss 对 `<pad>` 位置使用 `ignore_index`
- 输出层和 embedding 层做了 `weight tying`

这个项目可以继续向两个方向扩展：

- 用更大的语料继续训练，观察 tokenizer 和生成质量如何变化
- 在这个骨架上继续补 `checkpoint resume`、更细的实验对照和更长上下文
