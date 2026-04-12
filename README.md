# DeepLearning 学习笔记与实验

这个仓库包含两个方向的内容：

- 深度学习实验项目：围绕 MNIST、CIFAR-10、字符级 Transformer 和子词级 GPT 构建可运行、可对比的 PyTorch 训练代码
- 学习笔记：记录模型原理、数学直觉和实现过程

## 项目概览

### MNIST 实验

项目路径：[projects/01-mnist-cnn-experiments/README.md](projects/01-mnist-cnn-experiments/README.md)

- `MLP baseline`：测试集准确率 `96.12%`
- `CNN improved`：测试集准确率 `99.47%`
- 重点：从全连接基线升级到卷积模型，并完成训练流程模块化、结果落盘和可视化输出

### CIFAR-10 CNN 实验

项目路径：[projects/02-cifar10-cnn-experiments/README.md](projects/02-cifar10-cnn-experiments/README.md)

- `baseline`：测试集准确率 `73.25%`
- `improved`：测试集准确率 `87.35%`
- `resnet`：测试集准确率 `95.33%`
- 重点：沿着简单 CNN、工程化优化版、残差网络三条线做出清晰的性能迭代

### 字符级 Transformer 实验（Character Transformer）

项目路径：[projects/03-char-transformer-experiments/README.md](projects/03-char-transformer-experiments/README.md)

- `bigram`：最小字符级 next-token baseline
- `transformer`：带位置编码、因果掩码、多头注意力和 FFN 的 decoder-only Transformer
- 最新结果：`transformer v3` 验证集 `perplexity = 4.63`
- 重点：把自注意力笔记继续推进到可训练、可生成文本的最小语言模型实现，并支持不同 `temperature` 的生成对比

### 子词级 GPT 实验（Subword GPT）

项目路径：[projects/04-subword-gpt-experiments/README.md](projects/04-subword-gpt-experiments/README.md)

- `byte-level BPE tokenizer`：从原始文本自动学习 merge 规则
- `subword GPT`：带 special tokens、padding mask、weight tying 的 decoder-only GPT
- 最新结果：`subword-gpt v1` 验证集 `perplexity = 19.51`
- 重点：把字符级教学版 Transformer 继续推进到更像真实 LLM 训练流程的 tokenizer + GPT 项目

## 语言模型对比

下面这张表把当前仓库中两条语言模型主线并排放在一起。需要注意：

- `char-transformer v3` 和 `subword-gpt v1` 的 token 粒度不同
- 因此 `perplexity` 不能直接横向当作绝对优劣结论
- 更合适的理解方式是：它们分别代表“最小字符级教学版 Transformer”和“更接近真实 GPT 工作流的子词级模型”

| 模型                | 词元粒度（Tokenization） | 参数量    | 上下文长度 / `block_size` | 最佳验证损失 | 最佳验证困惑度 | 对外可展示的核心意义                                              |
| ------------------- | ------------------------ | --------: | -------------------------: | -----------: | -------------: | ----------------------------------------------------------------- |
| char-transformer v3 | character-level          | 2,286,593 |                        192 |       1.5333 |           4.63 | 字符级生成质量更强，能稳定学到对白格式、角色名和局部句式          |
| subword-gpt v1      | byte-level BPE           | 6,490,624 |                        160 |       2.9711 |          19.51 | 工程形态更接近真实 GPT，补齐 tokenizer、special tokens 和 padding |

当前阶段可以这样理解：

- 如果目标是解释 Transformer 机制和最小语言模型原理，`char-transformer v3` 更直观
- 如果目标是向真实 LLM 训练流程靠近，`subword-gpt v1` 是更重要的下一步
- 这两个项目不是互相替代，而是前后衔接的两站

## 笔记目录

笔记索引见 [notes/README.md](notes/README.md)。

- [notes/01-MLP与MNIST：从数据预处理到训练流程.md](notes/01-MLP与MNIST：从数据预处理到训练流程.md)：MLP 手写数字识别
- [notes/02-CNN与MNIST：从卷积直觉到图像分类实现.md](notes/02-CNN与MNIST：从卷积直觉到图像分类实现.md)：CNN 数学直觉与 MNIST 实战
- [notes/03-自注意力机制：从Q、K、V到缩放点积注意力.md](notes/03-自注意力机制：从Q、K、V到缩放点积注意力.md)：Transformer 自注意力机制推导
- [notes/04-Transformer语言模型：从位置编码到最小可训练实现.md](notes/04-Transformer语言模型：从位置编码到最小可训练实现.md)：从位置编码到最小 Transformer 语言模型实现
- [notes/05-子词级GPT：从BPE到更像真实LLM的训练流程.md](notes/05-子词级GPT：从BPE到更像真实LLM的训练流程.md)：从 byte-level BPE 到更像真实 GPT 的训练流程

## 快速开始

共享依赖文件位于 `projects/requirements.txt`。

### MNIST

```bash
cd projects/01-mnist-cnn-experiments
pip install -r ../requirements.txt
python train_mlp.py
python train_cnn.py
```

### CIFAR-10

```bash
cd projects/02-cifar10-cnn-experiments
pip install -r ../requirements.txt
python train_baseline.py
python train_improved.py
python train_resnet.py
```

### 字符级 Transformer

```bash
cd projects/03-char-transformer-experiments
pip install -r ../requirements.txt
python train_bigram.py
python train_transformer.py
python generate_samples.py --run-dir outputs/tinyshakespeare-transformer-v3 --temperatures 0.6 0.75 0.9
```

### 子词级 GPT

```bash
cd projects/04-subword-gpt-experiments
pip install -r ../requirements.txt
python train_gpt.py --experiment-name tinyshakespeare-subword-gpt-v1 --epochs 12 --steps-per-epoch 250 --eval-steps 50 --batch-size 12 --grad-accum-steps 2 --block-size 160 --min-sequence-length 48 --embedding-dim 256 --num-heads 8 --num-layers 8 --dropout 0.15 --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 150 --weight-decay 0.1 --grad-clip 1.0 --tokenizer-vocab-size 512 --device cuda --use-amp
python generate_samples.py --run-dir outputs/tinyshakespeare-subword-gpt-v1 --temperatures 0.6 0.8 1.0 --top-k 40 --top-p 0.95
```

说明：

- 四个实验项目现在都会默认把数据和输出写入各自项目目录
- 仍然可以通过 `--data-dir` 和 `--output-dir` 覆盖默认路径

## 仓库结构

```text
DeepLearning/
├─ notes/
├─ assets/
│  └─ images/
├─ projects/
│  ├─ requirements.txt
│  ├─ 01-mnist-cnn-experiments/
│  ├─ 02-cifar10-cnn-experiments/
│  ├─ 03-char-transformer-experiments/
│  └─ 04-subword-gpt-experiments/
└─ README.md
```

## 开源协议

本仓库中的代码、笔记与文档内容基于 [MIT License](./LICENSE) 开源。

补充说明：

- 本仓库的许可证仅覆盖当前仓库中自行编写和整理的训练代码、学习笔记、图示与文档结构。
- 数据集、论文内容、模型原始思想以及其他第三方原始资料，不因本仓库采用 MIT 协议而自动转授任何额外权利。
