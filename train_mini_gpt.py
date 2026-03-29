"""
Mini-GPT 训练脚本
用于训练唐诗/莎士比亚文本生成模型

使用方法:
    python train_mini_gpt.py --data_path ./data/tang_poetry.txt --epochs 50
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
import math

from mini_gpt import MiniGPT, CharTokenizer


# ============================================================================
# 数据集类
# ============================================================================

class TextDataset(Dataset):
    """
    文本数据集
    
    将长文本切分成固定长度的片段，用于训练。
    
    参数:
        text: 原始文本
        tokenizer: 分词器
        block_size: 上下文窗口大小（每个样本的长度）
    """
    def __init__(self, text, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 编码整个文本
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"数据集大小: {len(self.data)} 个字符")
        print(f"可生成样本数: {len(self)}")
        
    def __len__(self):
        # 计算可以切分出多少个完整的块
        # 如果数据太小，返回 0 而不是负数
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        """
        获取一个训练样本
        
        返回:
            x: 输入序列 [block_size]
            y: 目标序列 [block_size]（x 向后平移一位）
        """
        # 取出一个长度为 block_size + 1 的片段
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # 如果片段长度不够（边界情况），进行填充
        if len(chunk) < self.block_size + 1:
            # 用 0 填充（通常 0 对应特殊字符）
            padding = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        
        # 输入是前 block_size 个字符
        x = chunk[:-1]
        
        # 目标是后 block_size 个字符（向后平移一位）
        y = chunk[1:]
        
        return x, y


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    训练一个 epoch
    参数:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备（CPU 或 GPU）
        epoch: 当前 epoch 数
    
    返回:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    # 使用 tqdm 显示进度条，设置为简洁模式
    # leave=False: 完成后清除进度条，避免重复打印
    # mininterval=1.0: 最少每秒更新一次，减少刷新频率
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100, leave=False, mininterval=1.0)
    
    for batch_idx, (x, y) in enumerate(pbar):
        # 将数据移到设备上
        x = x.to(device)
        y = y.to(device)
        
        # 前向传播
        logits = model(x)  # [batch_size, seq_len, vocab_size]
        
        # 计算损失
        # 需要将 logits 和 y 展平成 2D 和 1D
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        y_flat = y.view(batch_size * seq_len)
        
        loss = nn.functional.cross_entropy(logits_flat, y_flat)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        
        # 每 100 步更新一次进度条显示
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    评估模型
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    返回:
        avg_loss: 平均损失
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        y_flat = y.view(batch_size * seq_len)
        
        loss = nn.functional.cross_entropy(logits_flat, y_flat)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def generate_samples(model, tokenizer, device, num_samples=3, max_length=100):
    """
    生成一些样本文本
    
    参数:
        model: 模型
        tokenizer: 分词器
        device: 设备
        num_samples: 生成样本数量
        max_length: 每个样本的最大长度
    """
    model.eval()
    
    # 一些可能的起始字符
    start_chars = ["春", "月", "风", "花", "山", "水"]
    
    print("\n" + "=" * 60)
    print("生成样本:")
    print("=" * 60)
    
    for i in range(num_samples):
        # 随机选择一个起始字符
        if i < len(start_chars):
            start_char = start_chars[i]
        else:
            start_char = start_chars[0]
        
        # 编码起始字符
        # tokenizer.encode 返回列表，例如 [10]
        # 我们需要形状 [batch_size=1, seq_len=1]
        start_ids = tokenizer.encode(start_char)  # [10]
        if len(start_ids) == 0:
            print(f"警告: 字符 '{start_char}' 无法编码，跳过")
            continue
        # 创建形状 [1, len(start_ids)] 的张量
        start_tokens = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 生成
        generated = model.generate(
            start_tokens,
            max_new_tokens=max_length,
            temperature=0.8,  # 稍微降低温度，让生成更连贯
            top_k=20
        )
        
        # 解码
        generated_text = tokenizer.decode(generated[0])
        
        print(f"\n样本 {i+1}:")
        print(generated_text)
    
    print("=" * 60 + "\n")


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练 Mini-GPT')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='训练文本文件路径')
    parser.add_argument('--block_size', type=int, default=128,
                        help='上下文窗口大小（默认: 128）')
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型维度（默认: 256）')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformer 层数（默认: 6）')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数（默认: 8）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小（默认: 32）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数（默认: 50）')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率（默认: 3e-4）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 概率（默认: 0.1）')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录（默认: ./checkpoints）')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='每隔多少个 epoch 评估一次（默认: 5）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"文本长度: {len(text)} 个字符")
    print(f"文本预览:\n{text[:200]}...\n")
    
    # ========================================================================
    # 2. 构建分词器
    # ========================================================================
    print("=" * 60)
    print("构建分词器...")
    print("=" * 60)
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    
    # ========================================================================
    # 3. 划分训练集和验证集
    # ========================================================================
    split_idx = int(len(text) * 0.9)  # 90% 训练，10% 验证
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"\n训练集大小: {len(train_text)} 个字符")
    print(f"验证集大小: {len(val_text)} 个字符")
    
    # ========================================================================
    # 4. 创建数据集和数据加载器
    # ========================================================================
    print("\n" + "=" * 60)
    print("创建数据集...")
    print("=" * 60)
    
    train_dataset = TextDataset(train_text, tokenizer, args.block_size)
    val_dataset = TextDataset(val_text, tokenizer, args.block_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # 如果在 Windows 上可能需要设为 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # ========================================================================
    # 5. 创建模型
    # ========================================================================
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)
    
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.block_size,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # ========================================================================
    # 6. 创建优化器
    # ========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # ========================================================================
    # 7. 训练循环
    # ========================================================================
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 定期评估
        if epoch % args.eval_interval == 0:
            val_loss, perplexity = evaluate(model, val_loader, device)
            print(f"\n[Epoch {epoch}/{args.epochs}] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 困惑度: {perplexity:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 生成样本
            generate_samples(model, tokenizer, device)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'tokenizer_char_to_id': tokenizer.char_to_id,
                    'tokenizer_id_to_char': tokenizer.id_to_char,
                    'args': args
                }, checkpoint_path)
                print(f"✓ 保存最佳模型到 {checkpoint_path}")
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点（静默保存）
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'tokenizer_char_to_id': tokenizer.char_to_id,
                'tokenizer_id_to_char': tokenizer.id_to_char,
                'args': args
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 最终评估
    val_loss, perplexity = evaluate(model, val_loader, device)
    print(f"\n最终验证损失: {val_loss:.4f}")
    print(f"最终困惑度: {perplexity:.2f}")
    
    # 生成最终样本
    generate_samples(model, tokenizer, device, num_samples=5, max_length=200)


if __name__ == "__main__":
    main()
