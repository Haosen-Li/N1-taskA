# 从零实现并验证小型 Transformer 语言模型

一个可实际训练的 decoder-only 自回归语言模型，词表为 20 个离散符号，训练语料是在线生成的键值记忆语言（不是自然语言）。核心网络、loss 和优化器都是手写展开的；PyTorch 只负责张量运算、参数管理和自动微分。

## 运行

```bash
cd /mnt/workspace/Wilson/N1/taska
source /mnt/workspace/Wilson/miniconda3/etc/profile.d/conda.sh && conda activate gps


python -m pytest -q test_core.py                   

CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --device cuda:1

python evaluate.py --checkpoint runs/direct/model.pt --device cuda:1
```

## 文件

| 文件 | 内容 |
|---|---|
| `model.py` | 手写的 Linear、LayerNorm、softmax、GELU、因果多头注意力、MLP、交叉熵、AdamW、梯度裁剪 |
| `task.py` | 词表、在线随机任务生成器、配对干预 |
| `train.py` | 训练循环、学习率调度、独立验证、保存权重和日志 |
| `evaluate.py` | 独立测试、上下文干预、长度泛化、Wilson 置信区间、绘图 |
| `test_core.py` | 与参考实现对照、有限差分、因果遮罩、数据对齐检查 |
| `WALKTHROUGH.md` | 从 token 到 optimizer 的公式与关键代码讲解 |
| `REPORT.md` | 实验结果、证据与局限 |
| `FOLLOWUP.md` | 补充实验：复现核验、多 seed、单变量消融、按绝对位置寻址的机制与因果干预、真实文本 LM、改变量时的预测 |
| `followup/` | 补充实验的脚本（外部替换变体，不改动核心代码）、34 次补充训练的权重（`runs/`）、结果 JSON（`results/`）、图（`figures/`） |

交付权重是 `runs/direct/model.pt`：2 层、4 头、隐藏宽度 64、119,040 参数、可学习绝对位置编码，另有固定的距离偏置（slope `[0,0,0.5,1]`）。同一目录下还有 `config.json`（参数与环境）、`first_backward.json`（第一次反传时每个参数的形状和梯度范数）、`results.json`（测试结果）和两张图。checkpoint 只保存模型与配置，不含优化器和 RNG 状态，所以要复现就从头训练。每步训练日志 `history.jsonl` 不随仓库发布，用上面的训练命令即可逐位重新生成（训练曲线见 `training.png`）。

## 任务为何必须使用远处上下文

```text
<bos> <store> K2 V5 <store> K7 V1 <store> K0 V6 <store> K4 V3
      V2 V5 V1 ...（32 / 48 / 64 个独立随机的干扰值）...
      <query> K7   →  预测 V1
```

每个样本独立随机选 4 个不同的键，每个键随机分配 8 种值之一，再随机查询其中一个键。干扰 token 与答案来自同一个值词表，但和标签相互独立。输入 `x` 到查询键 `K7` 为止，答案 `V1` 只出现在标签 `y` 中。训练时正确源值与答案相隔 35–76 个 token。

只看后缀（不覆盖记忆区）的任何预测器，期望准确率都是 1/8 = 12.5%。更强的检查是固定整个后缀、只遍历远处源值的 8 种取值：只看后缀的模型对这 8 个输入只能给出同一个答案，真正检索远处信息的模型则能 8 个全对。全部验收条件和结果见 `REPORT.md`。

词表第 3 号 token `<unused>` 是早期 `<query> K <answer> V` 格式留下的。保留它是为了让词表大小保持 20，从而参数形状和初始化随机数流都不变，交付权重才能逐位复现。
