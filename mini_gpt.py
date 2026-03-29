"""
Mini-GPT: 从零实现 Transformer 用于唐诗/莎士比亚文本生成
教学版本 - 包含详细注释，便于学习理解

作者：基于 Transformer 原理的教学实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 第一阶段：核心算子 (The Core)
# ============================================================================

class SelfAttention(nn.Module):
    """
    自注意力机制 (Self-Attention)
    
    核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    参数:
        d_model: 模型的特征维度
        d_k: Query 和 Key 的维度（通常等于 d_model）
        causal: 是否使用因果掩码（Decoder 需要，防止看到未来信息）
    """
    def __init__(self, d_model, d_k=None, causal=True):
        super().__init__()
        self.d_k = d_k if d_k is not None else d_model
        self.causal = causal
        
        # 定义 Q, K, V 的线性变换矩阵
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            attention_output: 注意力输出，形状 [batch_size, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 步骤 1: 计算 Q, K, V
        # Q, K, V 的形状都是 [batch_size, seq_len, d_k]
        Q = self.W_q(x)  # Query: "我要查询什么？"
        K = self.W_k(x)  # Key: "我能提供什么信息？"
        V = self.W_v(x)  # Value: "我的具体内容是什么？"
        
        # 步骤 2: 计算注意力分数 QK^T / sqrt(d_k)
        # [batch_size, seq_len, d_k] @ [batch_size, d_k, seq_len] 
        # -> [batch_size, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 步骤 3: 应用因果掩码（Causal Mask）- Decoder 专用
        # 防止位置 i 看到位置 j > i 的信息（不能看到未来）
        if self.causal:
            # 创建下三角掩码矩阵
            # [[1, 0, 0],
            #  [1, 1, 0],
            #  [1, 1, 1]]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            # 将上三角部分设为 -inf，softmax 后会变成 0
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 步骤 4: Softmax 归一化，得到注意力权重
        # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 步骤 5: 加权求和 Value
        # [batch_size, seq_len, seq_len] @ [batch_size, seq_len, d_k]
        # -> [batch_size, seq_len, d_k]
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    将特征空间拆分成多个"头"，每个头独立做注意力，最后拼接起来。
    这样可以让模型从不同的表示子空间学习信息。
    
    参数:
        d_model: 模型的特征维度
        num_heads: 注意力头的数量
        causal: 是否使用因果掩码
    """
    def __init__(self, d_model, num_heads=8, causal=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.causal = causal
        
        # 定义 Q, K, V 的线性变换（一次性为所有头计算）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output: 多头注意力输出，形状 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 步骤 1: 线性变换得到 Q, K, V
        # 形状: [batch_size, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 步骤 2: 拆分成多个头
        # 从 [batch_size, seq_len, d_model] 
        # 变成 [batch_size, seq_len, num_heads, d_k]
        # 再转置为 [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 步骤 3: 计算注意力分数
        # [batch_size, num_heads, seq_len, d_k] @ [batch_size, num_heads, d_k, seq_len]
        # -> [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 步骤 4: 应用因果掩码
        if self.causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 步骤 5: Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 步骤 6: 加权求和
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, d_k]
        # -> [batch_size, num_heads, seq_len, d_k]
        attention_output = torch.matmul(attention_weights, V)
        
        # 步骤 7: 拼接所有头
        # 转置回 [batch_size, seq_len, num_heads, d_k]
        # 再合并为 [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        # 步骤 8: 输出投影
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed Forward Network)
    
    简单的两层全连接网络，中间加激活函数。
    通常会先升维再降维，增加模型的表达能力。
    
    结构: Linear -> GELU -> Linear
    
    参数:
        d_model: 输入和输出的维度
        d_ff: 中间层的维度（通常是 d_model 的 4 倍）
        dropout: Dropout 概率
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 默认扩大 4 倍
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output: 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 升维 + GELU 激活
        x = self.linear1(x)
        x = F.gelu(x)  # GELU 比 ReLU 更平滑，效果通常更好
        x = self.dropout(x)
        
        # 降维
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


# ============================================================================
# 第二阶段：位置编码与模块组装 (The Architecture)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)
    
    Transformer 没有循环结构，无法感知位置信息。
    需要手动添加位置编码，告诉模型每个 token 的位置。
    
    这里实现原论文的正余弦位置编码：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        d_model: 模型的特征维度
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 分母项 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 偶数维度用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与训练，但会保存到模型中）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output: 加上位置编码后的张量，形状不变
        """
        seq_len = x.size(1)
        # 将位置编码加到输入上
        x = x + self.pe[:, :seq_len, :]
        return x


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码 (Learnable Positional Encoding)
    
    更简单的实现方式：直接用 Embedding 层学习位置编码。
    很多现代模型（如 GPT）都采用这种方式。
    
    参数:
        d_model: 模型的特征维度
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output: 加上位置编码后的张量，形状不变
        """
        batch_size, seq_len, d_model = x.shape
        
        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # 获取位置编码并加到输入上
        position_encodings = self.position_embeddings(positions)
        x = x + position_encodings
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 块 (Transformer Block)
    
    组合多头注意力和前馈网络，加上残差连接和层归一化。
    
    结构（Pre-LN 版本，更稳定）:
        x -> LayerNorm -> MultiHeadAttention -> Add(x) 
          -> LayerNorm -> FeedForward -> Add
    
    参数:
        d_model: 模型的特征维度
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        dropout: Dropout 概率
    """
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.attention = MultiHeadAttention(d_model, num_heads, causal=True)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 两个层归一化
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output: 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 第一个子层：多头注意力 + 残差连接
        # Pre-LN: 先归一化再做注意力
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)  # 残差连接
        
        # 第二个子层：前馈网络 + 残差连接
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)  # 残差连接
        
        return x


# ============================================================================
# 第三阶段：完整模型 (The Complete Model)
# ============================================================================

class MiniGPT(nn.Module):
    """
    Mini-GPT: 完整的 Transformer 语言模型
    
    用于自回归文本生成（给定前文，预测下一个字）。
    
    参数:
        vocab_size: 词表大小（字符数量）
        d_model: 模型的特征维度
        num_layers: Transformer 块的层数
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        max_len: 最大序列长度
        dropout: Dropout 概率
        use_learnable_pe: 是否使用可学习的位置编码
    """
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, 
                 d_ff=None, max_len=512, dropout=0.1, use_learnable_pe=True):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token Embedding: 将字符 ID 映射为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        if use_learnable_pe:
            self.positional_encoding = LearnablePositionalEncoding(d_model, max_len)
        else:
            self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠多个 Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最后的层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # 输出层：将特征映射回词表大小
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        初始化模型权重
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入的 token IDs，形状 [batch_size, seq_len]
        
        返回:
            logits: 每个位置对词表的预测分数，形状 [batch_size, seq_len, vocab_size]
        """
        # 步骤 1: Token Embedding
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.token_embedding(x)
        
        # 步骤 2: 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 步骤 3: 通过所有 Transformer 块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 步骤 4: 最后的层归一化
        x = self.ln_final(x)
        
        # 步骤 5: 投影到词表
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        logits = self.output_projection(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        自回归生成文本
        
        参数:
            start_tokens: 起始 token IDs，形状 [batch_size, seq_len] 或 [seq_len]
            max_new_tokens: 最多生成多少个新 token
            temperature: 温度参数，越大越随机，越小越确定
            top_k: 只从概率最高的 k 个 token 中采样（可选）
        
        返回:
            generated: 生成的完整序列，形状 [batch_size, seq_len + max_new_tokens]
        """
        self.eval()  # 切换到评估模式
        
        # 确保是 2D 张量
        if start_tokens.dim() == 1:
            start_tokens = start_tokens.unsqueeze(0)
        
        generated = start_tokens
        
        for _ in range(max_new_tokens):
            # 截断序列到最大长度，避免超出位置编码范围
            if generated.size(1) > self.max_len:
                generated = generated[:, -self.max_len:]
            
            # 获取当前序列的 logits
            # 注意：只取最后一个位置的预测
            logits = self.forward(generated)  # [batch_size, seq_len, vocab_size]
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 应用温度
            logits = logits / temperature
            
            # Top-k 采样（可选）
            if top_k is not None:
                # 保留 top-k，其余设为 -inf
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_k_logits[:, -1:]] = float('-inf')
            
            # Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 检查概率是否有效
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("警告: 检测到无效概率，使用均匀分布")
                probs = torch.ones_like(probs) / probs.size(-1)
            
            # 从概率分布中采样
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # 验证 token ID 是否在有效范围内
            if (next_token >= self.vocab_size).any() or (next_token < 0).any():
                print(f"警告: 生成了无效的 token ID: {next_token.item()}, vocab_size: {self.vocab_size}")
                next_token = torch.clamp(next_token, 0, self.vocab_size - 1)
            
            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================================================
# 辅助函数：简单的字符级分词器
# ============================================================================

class CharTokenizer:
    """
    字符级分词器
    
    将文本中的每个字符映射为一个唯一的 ID。
    适用于中文（唐诗）和英文（莎士比亚）。
    """
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, text):
        """
        从文本中构建词表
        
        参数:
            text: 训练文本（字符串）
        """
        # 获取所有唯一字符并排序
        unique_chars = sorted(set(text))
        
        # 构建映射
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        
        print(f"词表大小: {self.vocab_size}")
        print(f"示例字符: {unique_chars[:20]}")
        
    def encode(self, text):
        """
        将文本编码为 ID 序列
        
        参数:
            text: 输入文本
        
        返回:
            ids: ID 列表
        """
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
    
    def decode(self, ids):
        """
        将 ID 序列解码为文本
        
        参数:
            ids: ID 列表或张量
        
        返回:
            text: 解码后的文本
        """
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return ''.join([self.id_to_char.get(i, '') for i in ids])


# ============================================================================
# 示例：如何使用这个模型
# ============================================================================

if __name__ == "__main__":
    # 演示模型的基本使用
    
    print("=" * 60)
    print("Mini-GPT 演示")
    print("=" * 60)
    
    # 1. 创建一个简单的分词器
    sample_text = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(sample_text)
    
    print(f"\n原始文本: {sample_text}")
    
    # 2. 编码文本
    encoded = tokenizer.encode(sample_text)
    print(f"编码后: {encoded[:20]}...")
    
    # 3. 解码回文本
    decoded = tokenizer.decode(encoded)
    print(f"解码后: {decoded}")
    
    # 4. 创建模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,          # 较小的模型用于演示
        num_layers=4,         # 4 层 Transformer
        num_heads=4,          # 4 个注意力头
        max_len=256,          # 最大序列长度
        dropout=0.1
    )
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 前向传播测试
    test_input = torch.tensor([encoded[:10]])  # 取前 10 个字符
    print(f"\n测试输入形状: {test_input.shape}")
    
    with torch.no_grad():
        logits = model(test_input)
        print(f"输出 logits 形状: {logits.shape}")
        print(f"预期形状: [batch_size=1, seq_len=10, vocab_size={tokenizer.vocab_size}]")
    
    # 6. 生成测试
    print("\n" + "=" * 60)
    print("生成测试（随机初始化的模型）")
    print("=" * 60)
    
    start_text = "春"
    start_tokens = torch.tensor([tokenizer.encode(start_text)])
    
    generated = model.generate(
        start_tokens, 
        max_new_tokens=20,
        temperature=1.0,
        top_k=10
    )
    
    generated_text = tokenizer.decode(generated[0])
    print(f"起始文本: {start_text}")
    print(f"生成文本: {generated_text}")
    print("\n注意: 模型未训练，生成的是随机文本。")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
