# Mini-GPT: 从零实现 Transformer 文本生成器

这是一个教学项目，从 `nn.Linear` 开始手工实现完整的 Transformer 模型，用于唐诗/莎士比亚文本自动生成。

## 📚 项目结构

```
.
├── mini_gpt.py           # 核心模型实现（所有 Transformer 组件）
├── train_mini_gpt.py     # 训练脚本
├── generate_text.py      # 文本生成脚本
├── prepare_data.py       # 数据准备脚本
└── README_MINI_GPT.md    # 本文件
```

## 🎯 学习路线

### 第一阶段：核心算子 (The Core)

在 `mini_gpt.py` 中实现：

1. **SelfAttention**: 自注意力机制
   - 手写 Q, K, V 矩阵乘法
   - 实现缩放点积注意力公式
   - 添加因果掩码（Causal Mask）

2. **MultiHeadAttention**: 多头注意力
   - 将特征空间拆分成多个头
   - 并行计算多个注意力
   - 拼接并投影输出

3. **FeedForward**: 前馈神经网络
   - 两层全连接网络
   - GELU 激活函数

### 第二阶段：位置编码与模块组装 (The Architecture)

4. **PositionalEncoding**: 正余弦位置编码
   - 实现原论文的位置编码公式
   - 或使用可学习的位置 Embedding

5. **TransformerBlock**: Transformer 块
   - 组合注意力和前馈网络
   - 添加残差连接
   - Pre-LN 层归一化

6. **MiniGPT**: 完整模型
   - Token Embedding
   - 堆叠多层 Transformer
   - 输出投影层

### 第三阶段：数据流与训练 (The Pipeline)

7. **CharTokenizer**: 字符级分词器
   - 简单的字符到 ID 映射

8. **TextDataset**: 文本数据集
   - 构造上下文窗口
   - 自回归训练样本

9. **generate()**: 自回归生成
   - 逐个预测下一个字符
   - 支持温度和 Top-k 采样

## 🚀 快速开始

### 1. 准备数据

**下载莎士比亚文本:**
```bash
python prepare_data.py --type shakespeare --output ./data/shakespeare.txt
```

**创建唐诗示例数据:**
```bash
python prepare_data.py --type tang_poetry --output ./data/tang_poetry.txt
```

### 2. 训练模型

**训练莎士比亚模型（小型）:**
```bash
python train_mini_gpt.py \
    --data_path ./data/shakespeare.txt \
    --block_size 128 \
    --d_model 256 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 32 \
    --epochs 50 \
    --lr 3e-4 \
    --save_dir ./checkpoints/shakespeare
```

**训练唐诗模型（小型）:**
```bash
python train_mini_gpt.py \
    --data_path ./data/tang_poetry.txt \
    --block_size 64 \
    --d_model 128 \
    --num_layers 4 \
    --num_heads 4 \
    --batch_size 16 \
    --epochs 100 \
    --lr 3e-4 \
    --save_dir ./checkpoints/tang_poetry
```

**训练更大的模型（如果有 GPU）:**
```bash
python train_mini_gpt.py \
    --data_path ./data/shakespeare.txt \
    --block_size 256 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --batch_size 64 \
    --epochs 100 \
    --lr 3e-4 \
    --save_dir ./checkpoints/shakespeare_large
```

### 3. 生成文本

**批量生成:**
```bash
python generate_text.py \
    --checkpoint ./checkpoints/shakespeare/best_model.pt \
    --prompt "To be or not to be" \
    --length 200 \
    --temperature 0.8 \
    --top_k 20 \
    --num_samples 3
```

**交互式生成:**
```bash
python generate_text.py \
    --checkpoint ./checkpoints/tang_poetry/best_model.pt
```

### 4. 测试模型（不训练）

直接运行 `mini_gpt.py` 查看模型演示：
```bash
python mini_gpt.py
```

## 📊 模型参数说明

| 参数 | 说明 | 推荐值（小型） | 推荐值（大型） |
|------|------|---------------|---------------|
| `d_model` | 模型维度 | 128-256 | 512-768 |
| `num_layers` | Transformer 层数 | 4-6 | 8-12 |
| `num_heads` | 注意力头数 | 4-8 | 8-16 |
| `block_size` | 上下文窗口 | 64-128 | 256-512 |
| `batch_size` | 批次大小 | 16-32 | 64-128 |
| `lr` | 学习率 | 3e-4 | 1e-4 - 3e-4 |

## 🔍 代码学习要点

### 1. 注意力机制的核心公式

```python
# 在 SelfAttention.forward() 中
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
attention_weights = F.softmax(attention_scores, dim=-1)
attention_output = torch.matmul(attention_weights, V)
```

### 2. 因果掩码（防止看到未来）

```python
# 创建下三角掩码
mask = torch.tril(torch.ones(seq_len, seq_len))
attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
```

### 3. 多头注意力的拆分与拼接

```python
# 拆分: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

# 拼接: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
```

### 4. 残差连接与层归一化（Pre-LN）

```python
# Pre-LN: 先归一化，再做注意力，最后残差连接
attn_output = self.attention(self.ln1(x))
x = x + self.dropout(attn_output)
```

### 5. 自回归生成

```python
# 每次只预测下一个 token
logits = model(generated)[:, -1, :]  # 只取最后一个位置
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated = torch.cat([generated, next_token], dim=1)
```

## 📈 训练技巧

1. **从小模型开始**: 先用小参数量验证代码正确性
2. **监控困惑度**: 困惑度下降说明模型在学习
3. **调整温度**: 生成时温度越低越保守，越高越随机
4. **Top-k 采样**: 限制采样范围，避免生成低质量 token
5. **梯度裁剪**: 防止梯度爆炸（已在代码中实现）
6. **学习率调度**: 使用余弦退火逐渐降低学习率

## 🎓 学习建议

1. **逐行阅读代码**: 每个函数都有详细注释
2. **打印张量形状**: 在关键位置添加 `print(tensor.shape)` 理解数据流
3. **可视化注意力**: 可以保存 `attention_weights` 并可视化
4. **实验不同参数**: 观察模型大小对效果的影响
5. **对比原论文**: 阅读 "Attention is All You Need" 论文

## 🐛 常见问题

**Q: 训练很慢怎么办？**
- 减小 `batch_size`、`d_model`、`num_layers`
- 使用 GPU（代码会自动检测）
- 减少 `block_size`

**Q: 生成的文本质量不好？**
- 增加训练数据量
- 训练更多 epochs
- 调整 `temperature` 和 `top_k`
- 使用更大的模型

**Q: 显存不足？**
- 减小 `batch_size`
- 减小 `block_size`
- 减小 `d_model`

**Q: 如何获取更多唐诗数据？**
- https://github.com/chinese-poetry/chinese-poetry
- https://github.com/jackeyGao/chinese-poetry

## 📖 参考资料

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy 的简化 GPT 实现

## 📝 许可

本项目仅用于教学目的。

---

**祝学习愉快！🎉**
