# Notes Assistant SFT 实验速查

这个目录承接基于预训练模型的轻量微调主线，使用现有中文笔记构建领域问答数据，并在小型指令模型上完成 `SFT + LoRA/QLoRA + 评测 + Demo` 的完整流程。

## 关联笔记

- [07-指令微调与LoRA：从预训练模型到领域助教](../../notes/07-指令微调与LoRA：从预训练模型到领域助教.md)

## 实验内容

| 实验 | 作用 | 代表结果 |
| --- | --- | --- |
| `notes-assistant-qwen25-0p5b` | 标准训练配置 | 字符级 F1 `0.285 -> 0.444`，`28/30` 样本优于基座模型 |
| `notes-assistant-qwen25-0p5b-smoke` | 链路冒烟验证 | 字符级 F1 `0.331 -> 0.331`，仅用于验证链路可运行 |

## 代表结果

正式评测显示，平均字符级 F1 从 `0.285` 提升到 `0.444`，`30` 道 held-out 测试题里有 `28` 道微调后回答优于基座模型。更完整的结果解读和代表问答样例放在对应的第 `07` 章笔记中，这里只保留实验速查所需的结果摘要和配图。

<p align="center">
  <img src="../../assets/showcase/notes-assistant-qwen25-0p5b-public-results.png" alt="Notes Assistant 公开代表结果" width="920" />
</p>

## 运行命令

先安装共享依赖：

```bash
pip install -r ../requirements.txt
```

生成本地数据集：

```bash
python prepare_dataset.py --overwrite
```

运行完整 SFT：

```bash
python train_sft.py --experiment-name notes-assistant-qwen25-0p5b
```

小规模冒烟命令：

```bash
python train_sft.py --smoke
```

做基座模型 vs 微调模型对比评测：

```bash
python evaluate_qa.py --run-dir outputs/notes-assistant-qwen25-0p5b
```

启动本地 Demo：

```bash
python launch_demo.py --run-dir outputs/notes-assistant-qwen25-0p5b
```

## 常用覆盖参数

```bash
python train_sft.py ^
  --experiment-name notes-assistant-dev ^
  --max-seq-length 512 ^
  --batch-size 1 ^
  --grad-accum-steps 16 ^
  --learning-rate 2e-4 ^
  --epochs 3 ^
  --lora-r 16 ^
  --lora-alpha 32
```

在未准备 `bitsandbytes` / `4-bit` 量化环境时，可先切到普通 LoRA 验证链路：

```bash
python train_sft.py --quantization-mode none --smoke
```

## 输出目录

- `data/notes-assistant-qa.jsonl`：自动生成的指令数据
- `data/notes-assistant-dataset-summary.json`：数据集统计和章节切分摘要
- `outputs/<experiment-name>/adapter/`：LoRA adapter 权重与 tokenizer
- `outputs/<experiment-name>/metrics.json`：训练和验证指标
- `outputs/<experiment-name>/samples.md`：测试样例生成结果
- `outputs/<experiment-name>/evaluation/`：评测报告、预测明细和人工打分模板

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `prepare_dataset.py` | 从现有笔记自动生成 JSONL 指令数据 |
| `train_sft.py` | SFT 训练入口 |
| `evaluate_qa.py` | 基座模型与 adapter 对比评测入口 |
| `launch_demo.py` | Gradio Demo 入口 |
| `notes_assistant_experiments/dataset_builder.py` | 章节解析、切分和样本生成 |
| `notes_assistant_experiments/train.py` | LoRA/QLoRA 训练主流程 |
| `notes_assistant_experiments/evaluation.py` | 自动评测与人工评审表导出 |

## 当前边界

- v1 只做 `SFT + LoRA/QLoRA`，不做 `RAG`、多轮记忆和工具调用。
- 数据只来自当前仓库的课程笔记，因此更适合作为领域学习助教，而不是通用聊天模型。
- 这条实验线的重点是补齐真实 LLM 工程工作流，而不是追求最强效果。
