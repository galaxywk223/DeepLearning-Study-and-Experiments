# 字符级 Transformer 实验速查

主阅读入口：

- [04-自注意力机制：从Q、K、V到缩放点积注意力](../../notes/04-自注意力机制：从Q、K、V到缩放点积注意力.md)
- [05-Transformer语言模型：从位置编码到最小可训练实现](../../notes/05-Transformer语言模型：从位置编码到最小可训练实现.md)

## 包含实验

| 实验 | 作用 | 最佳验证困惑度 |
| --- | --- | ---: |
| `bigram` | 建立最小字符级 next-token 基线 | `12.73` |
| `transformer` | 补齐最小 decoder-only Transformer 骨架 | `8.32` |
| `transformer v2` | 扩大上下文和模型容量 | `5.33` |
| `transformer v3` | 当前仓库里更成熟的字符级生成结果 | `4.63` |

![Character Transformer v3 loss curve](../../assets/showcase/char-transformer-v3-loss-curve.png)

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

这些目录默认只用于本地运行，不纳入版本控制。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `train_bigram.py` | bigram 基线入口 |
| `train_transformer.py` | Transformer 训练入口 |
| `generate_samples.py` | 采样导出入口 |
| `char_transformer_experiments/models.py` | 模型定义 |
| `char_transformer_experiments/runner.py` | 训练主流程 |
