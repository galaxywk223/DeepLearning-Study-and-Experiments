# 字符级 Transformer 实验速查

这个目录承接字符级语言模型主线，从 `bigram` 基线开始，逐步补齐最小 decoder-only Transformer 的训练与采样流程。

## 关联笔记

- [04-自注意力机制：从Q、K、V到缩放点积注意力](../../notes/04-自注意力机制：从Q、K、V到缩放点积注意力.md)
- [05-Transformer语言模型：从位置编码到最小可训练实现](../../notes/05-Transformer语言模型：从位置编码到最小可训练实现.md)

## 实验内容

| 实验 | 作用 | 最佳验证困惑度 |
| --- | --- | ---: |
| `bigram` | 建立最小字符级 next-token 基线 | `12.73` |
| `transformer` | 补齐最小 decoder-only Transformer 骨架 | `8.32` |
| `transformer v2` | 扩大上下文和模型容量 | `5.33` |
| `transformer v3` | 当前仓库里更成熟的字符级生成结果 | `4.63` |

## 代表结果

收敛曲线用于观察模型容量和上下文长度扩大后，验证困惑度如何继续下降。

<p align="center">
  <img src="../../assets/showcase/char-transformer-v3-loss-curve.png" alt="Character Transformer v3 收敛曲线" width="760" />
</p>

## 运行命令

```bash
pip install -r ../requirements.txt
python train_bigram.py
python train_transformer.py
```

导出不同温度的生成样例：

```bash
python generate_samples.py --run-dir outputs/<experiment-name> --temperatures 0.6 0.75 0.9
```

## 输出目录

- `data/`：语料文件
- `outputs/<experiment-name>/`：配置、指标、最佳权重、生成样例和 loss 曲线
- 这些目录默认只用于本地运行，不纳入版本控制。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_bigram.py` | bigram 基线入口 |
| `train_transformer.py` | Transformer 训练入口 |
| `generate_samples.py` | 采样导出入口 |
| `char_transformer_experiments/models.py` | 模型定义 |
| `char_transformer_experiments/runner.py` | 训练主流程 |
