# token → embedding → attention → MLP → logits → loss → backward → optimizer

下面直接对应 `model.py`、`task.py` 和 `train.py`。批大小记为 B，序列长度 T，隐藏宽度 C=64，注意力头数 H=4，每头宽度 D=16，词表大小 V=20。

## 1. token：离散 ID，不是连续特征

`task.py` 的 `VOCAB` 为每个完整符号指定唯一整数。例如 K0 的 ID 是 4，V0 的 ID 是 12。数据生成器直接产出整数 ID，`decode` 只用于把 ID 还原成可读符号，没有借用现成 tokenizer。3 号 token `<unused>` 是早期格式留下的；保留它是为了让词表大小保持 20，交付权重才能逐位复现。

序列 `s` 必须右移一位构造监督：

```python
x = s[:, :-1]
y = s[:, 1:]
```

第 t 个输出是在看完 `x[:, :t+1]` 后预测 `y[:, t]`，也就是原序列的第 t+1 个 token。训练允许一次计算所有位置，因为因果遮罩严格禁止读取未来。最终答案只在 y 中，x 的最后一个 token 是查询键 K，不能把最终答案本身作为输入。

## 2. embedding：把 ID 变成可学习向量

```python
x = self.token_embedding[token_ids] * math.sqrt(self.cfg.d_model)
x = x + self.positions[:token_ids.shape[1]]
```

表 E 的形状是 `[V,C]`，整数查表得到 `[B,T,C]`。对 ID 不求导，但损失对选中的 E 行有梯度；相同 token 出现多次时这些梯度相加。初始 E 标准差为 0.02，乘 √C 增加内容向量相对于固定位置向量的初始尺度。

模型使用形状 `[256,64]`、标准差 0.02 初始化的**可学习位置表**，对应行与 token embedding 相加。早期实验用过固定正弦位置编码 `sin/cos(pos / 10000^(2i/C))`，当前代码只保留可学习位置表。没有任何位置信号的内容注意力难以稳定定位相邻键值；位置编码与下述距离偏置提供先后关系。训练未覆盖的可学习位置行没有得到数据梯度，所以更长序列上的表现必须另测，不能因为表有 256 行就宣称学会了 256 长度。

## 3. LayerNorm、残差与多头因果注意力

每层采用 pre-norm：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

LayerNorm 对每个 token 的 C 个特征单独计算均值/方差，再乘可学习 scale、加 bias。不能沿时间维归一化，否则当前位置可能通过均值看到未来。`eps` 避免方差太小时除零；方差采用总体方差而非无偏样本估计，与神经网络 LayerNorm 定义一致。残差给信息与梯度提供直接通路，pre-norm 有利于稳定训练。

注意力的线性映射和运算完全展开：

```python
q, k, v = self.qkv(x).chunk(3, dim=-1)
# reshape / transpose -> [B,H,T,D]
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
weights = softmax(scores.masked_fill(~allowed, float("-inf")))
mixed = weights @ v
```

Q 表示当前位置用来寻找信息的特征，K 表示各位置供匹配的特征，V 表示实际被搬运的信息。`QKᵀ` 得到 `[B,H,T,T]`，行是查询位置，列是来源位置。若分量近似独立、方差接近 1，点积方差随 D 增长，除 √D 能抑制 softmax 过早饱和。

遮罩允许列 j ≤ 行 i，**对角线保留**：当前位置的 token 本来就是已知输入。未来位置的分数在 softmax 前设为 −∞，使对应权重精确为 0。如果先 softmax 再清零，权重总和不再等于 1，且未来分数可能已经影响归一化分母。

softmax 先减去当前行最大值，再指数化、除以总和，避免指数上溢；减去常数不会改变分布。`weights @ v` 在时间维汇集过去信息。随后把各头拼接成 `[B,T,C]`，通过输出矩阵混合各头。`transpose` 后使用 `contiguous().reshape(...)` 明确内存排列与拼接意图。

正式模型还在分数中显式加入 `−slope[h] * max(i−j,0)`，四个头的固定 slope 为 `[0,0,0.5,1]`。前两个头不惩罚远距离，后两个头倾向较近的位置。这是类似 ALiBi 的距离先验，不是硬窗口，也没有读取标签或标记正确源位置；具体信息仍由 Q/K/V 学习。

本任务的键和值位于相邻但不同的位置。第一层可以在值的位置构造键值关联，下一层用末尾查询键检索对应值。这个描述只是结构允许的计算路径，**没有通过电路分析证明每个头实际承担了哪一步**；是否检索由干预实验判断。旧格式额外要求将查询键的信息传到 `<answer>` 位置，在本次尝试的训练预算内表现明显较差；由于同时改变了配置，不能单独量化标记的因果影响。

## 4. MLP：逐 token 的非线性计算

```python
return self.down(gelu(self.up(x)))
```

`[B,T,64] → [B,T,256] → [B,T,64]`，两次显式 `x @ weight + bias`，中间用手写 tanh 近似 GELU。扩大中间维度给非线性组合更多空间。没有非线性，两次线性映射可以合并成一次，表达能力不足。MLP 不在不同 token 间搬运信息，跨位置通信来自 attention。

## 5. logits：每个位置对整个词表打分

最终 LayerNorm 后，用独立的 `[C,V]` 矩阵得到 `[B,T,V]`。logits 是未归一化实数，不是概率。本实现没有绑定 embedding 与输出权重，以便清楚观察各自梯度。预测时读取最后一个位置的 logits 取 argmax，也就是单步 greedy decoding；多步生成就是把预测出的 token 拼回输入尾部再重复。本任务每条样本只需预测一个答案，所以代码里没有单独的 generate 函数，也不声称有生成自然语言的能力。

## 6. loss：稳定的 next-token 负对数似然

```python
correct = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
nll = torch.logsumexp(logits, dim=-1) - correct
```

这等于 `−log softmax(logits)[target]`，但避免先算非常小的概率再取 log。`gather` 只取每个位置正确标签对应的得分。`nll.mean()` 是常规全序列 next-token loss。

任务中几十个干扰值来自独立均匀随机数，对每个干扰位置，任何无泄漏模型的最优期望 loss 都是 `ln(8)≈2.079`。因此平均 loss 不会趋近 0，不能拿它直接衡量长距离检索。正式训练使用：

```python
loss = lm_loss + 2.0 * answer_loss
```

答案位置既在 LM 平均项里，也有额外权重。这样加强一个远距离监督信号，避免被大量不可预测的随机 token 稀释。它是**带答案加权的自回归训练目标**，不是未经修改的纯平均语言建模目标，也不是只计算答案的分类模型。两个分量都单独记录。

## 7. backward：真正沿整个计算图求导

```python
optimizer.zero_grad()
logits = model(batch.x)
loss, _, _ = objective(logits, batch.y, answer_weight)
loss.backward()
```

这是唯一委托给自动微分的核心步骤：没有手写反向引擎。前向的每个张量运算都会建立计算图，autograd 使用链式法则逆序传播，并把参数梯度放在 `.grad`。

关键局部导数可以直接解释：

- 单个交叉熵对 logits 的导数为 `p − one_hot(y)`，再乘对应的平均与答案加权系数。正确标签概率太小时，其 logit 梯度为负；梯度下降会把它提高。
- 对 `Y = XW + b`，有 `dW = Xᵀ dY`、`dX = dY Wᵀ`，批次和时间维贡献求和；`db` 也是求和。
- 对 `O = AV`，`dV = Aᵀ dO`、`dA = dO Vᵀ`。
- 对一行 `A = softmax(S)`，`dS = A * (dA − sum(dA * A))`。因果遮罩位置 A=0，没有通过未来位置的梯度。
- 对 `S = QKᵀ/√D`，`dQ = dS K/√D`，`dK = dSᵀ Q/√D`。
- 对残差 `Y = X + f(X)`，梯度为 `dX = dY + J_fᵀ dY`，包含一条直接传回的路径。

无需每步自己构造这些雅可比矩阵，autograd 执行对应的向量-雅可比乘积。`test_core.py` 对整块输入和选定参数做有限差分，对 loss 梯度做独立参考对照；首次实际训练的每个参数形状与梯度范数保存在 `first_backward.json`。清零必须在下一次反传前完成，否则 PyTorch 默认会累加梯度。

## 8. optimizer：梯度如何变成参数更新

先计算所有梯度的全局 L2 范数；超过 1 时统一缩放。这不改变梯度方向，限制一次异常更新的幅度。记录的是**裁剪前**范数。

手写 AdamW 对每个参数维护一阶矩 m 和二阶矩 v：

```text
m ← β1 m + (1−β1) g
v ← β2 v + (1−β2) g²
m_hat = m / (1−β1^t)
v_hat = v / (1−β2^t)
θ ← (1−lr·weight_decay) θ
θ ← θ − lr·m_hat / (sqrt(v_hat)+eps)
```

m 平滑梯度方向，v 调节各参数的有效步长；从零初始化会让早期矩估计偏小，所以进行偏差校正。AdamW 的衰减直接作用于参数，没有把衰减项加进自适应矩。为保持代码简单，本例对所有参数使用同一衰减率 0.01，包括 bias 和 LayerNorm 参数；生产实现常分组排除它们。

参数更新放在 `torch.no_grad()` 中，避免优化步骤又被记录到模型计算图里。学习率先 warmup 50 步，然后余弦下降至峰值的 10%。优化器数值测试用相同梯度运行多个更新，与 `torch.optim.AdamW` 对照；**只有测试使用参考优化器，训练完全使用手写版本**。

## 如何判断实现正确、训练有效

这两件事必须分别验证。数值与遮罩测试检查实现；独立样本、干预实验、长度测试检查学习行为。loss 下降不等于长距离检索成功，关注远处的注意力权重也不等于使用了远处信息。最终结论的边界见 `REPORT.md`。
