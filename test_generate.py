#!/usr/bin/env python3
"""
快速测试脚本：验证模型生成功能
"""

import torch
from mini_gpt import MiniGPT, CharTokenizer

def test_generation():
    """测试生成功能"""
    print("=" * 60)
    print("测试模型生成功能")
    print("=" * 60)
    
    # 创建简单的测试数据
    test_text = "abcdefghijklmnopqrstuvwxyz"
    
    # 创建分词器
    print("\n1. 创建分词器...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(test_text)
    print(f"   词表大小: {tokenizer.vocab_size}")
    
    # 创建小模型
    print("\n2. 创建模型...")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        max_len=128,
        dropout=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   设备: {device}")
    
    # 测试编码
    print("\n3. 测试编码...")
    start_char = "a"
    start_ids = tokenizer.encode(start_char)
    print(f"   字符 '{start_char}' -> IDs: {start_ids}")
    
    # 创建 start_tokens
    start_tokens = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(device)
    print(f"   start_tokens 形状: {start_tokens.shape}")
    print(f"   start_tokens 值: {start_tokens}")
    
    # 测试前向传播
    print("\n4. 测试前向传播...")
    model.eval()
    with torch.no_grad():
        logits = model.forward(start_tokens)
        print(f"   logits 形状: {logits.shape}")
        print(f"   期望形状: [1, {len(start_ids)}, {tokenizer.vocab_size}]")
    
    # 测试生成
    print("\n5. 测试生成...")
    try:
        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                max_new_tokens=10,
                temperature=1.0,
                top_k=5
            )
        print(f"   生成序列形状: {generated.shape}")
        print(f"   生成序列 IDs: {generated[0].tolist()}")
        
        # 解码
        generated_text = tokenizer.decode(generated[0])
        print(f"   生成文本: '{generated_text}'")
        
        print("\n✅ 测试通过！生成功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败！")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation()
    exit(0 if success else 1)
