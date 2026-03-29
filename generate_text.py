"""
使用训练好的 Mini-GPT 模型生成文本

使用方法:
    python generate_text.py --checkpoint ./checkpoints/best_model.pt --prompt "春" --length 200
"""

import torch
import argparse
from mini_gpt import MiniGPT, CharTokenizer


def load_model(checkpoint_path, device):
    """
    加载训练好的模型
    
    参数:
        checkpoint_path: 检查点文件路径
        device: 设备
    
    返回:
        model: 加载的模型
        tokenizer: 分词器
        args: 训练时的参数
    """
    print(f"加载模型从 {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重建分词器
    tokenizer = CharTokenizer()
    tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
    tokenizer.id_to_char = checkpoint['tokenizer_id_to_char']
    tokenizer.vocab_size = len(tokenizer.char_to_id)
    
    # 获取训练参数
    train_args = checkpoint['args']
    
    # 重建模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=train_args.d_model,
        num_layers=train_args.num_layers,
        num_heads=train_args.num_heads,
        max_len=train_args.block_size,
        dropout=0.0  # 生成时不需要 dropout
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    print(f"  - 词表大小: {tokenizer.vocab_size}")
    print(f"  - 模型维度: {train_args.d_model}")
    print(f"  - 层数: {train_args.num_layers}")
    print(f"  - 注意力头数: {train_args.num_heads}")
    
    if 'val_loss' in checkpoint:
        print(f"  - 验证损失: {checkpoint['val_loss']:.4f}")
    
    return model, tokenizer, train_args


def generate_text(model, tokenizer, prompt, max_length, temperature, top_k, device):
    """
    生成文本
    
    参数:
        model: 模型
        tokenizer: 分词器
        prompt: 起始文本
        max_length: 生成的最大长度
        temperature: 温度参数
        top_k: Top-k 采样参数
        device: 设备
    
    返回:
        generated_text: 生成的文本
    """
    # 编码起始文本
    start_tokens = tokenizer.encode(prompt)
    
    if len(start_tokens) == 0:
        print(f"警告: 起始文本 '{prompt}' 无法编码，使用词表中的第一个字符")
        start_tokens = [0]
    
    start_tokens = torch.tensor([start_tokens]).to(device)
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    # 解码
    generated_text = tokenizer.decode(generated[0])
    
    return generated_text


def interactive_mode(model, tokenizer, device):
    """
    交互式生成模式
    
    参数:
        model: 模型
        tokenizer: 分词器
        device: 设备
    """
    print("\n" + "=" * 60)
    print("交互式生成模式")
    print("=" * 60)
    print("输入起始文本，模型会自动续写。")
    print("输入 'quit' 或 'exit' 退出。")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # 获取用户输入
            prompt = input("起始文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式。")
                break
            
            if not prompt:
                print("请输入起始文本！")
                continue
            
            # 获取生成参数
            try:
                length = int(input("生成长度 (默认 100): ") or "100")
                temperature = float(input("温度 (默认 0.8): ") or "0.8")
                top_k = int(input("Top-k (默认 20): ") or "20")
            except ValueError:
                print("参数输入错误，使用默认值。")
                length = 100
                temperature = 0.8
                top_k = 20
            
            # 生成文本
            print("\n生成中...")
            generated_text = generate_text(
                model, tokenizer, prompt, length, temperature, top_k, device
            )
            
            print("\n" + "-" * 60)
            print("生成结果:")
            print("-" * 60)
            print(generated_text)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n退出交互模式。")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='使用 Mini-GPT 生成文本')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='起始文本（如果不提供，进入交互模式）')
    parser.add_argument('--length', type=int, default=200,
                        help='生成的最大长度（默认: 200）')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='温度参数，越大越随机（默认: 0.8）')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Top-k 采样参数（默认: 20）')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='生成样本数量（默认: 1）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model, tokenizer, train_args = load_model(args.checkpoint, device)
    
    # 如果没有提供 prompt，进入交互模式
    if args.prompt is None:
        interactive_mode(model, tokenizer, device)
    else:
        # 批量生成模式
        print("\n" + "=" * 60)
        print("生成文本")
        print("=" * 60)
        print(f"起始文本: {args.prompt}")
        print(f"生成长度: {args.length}")
        print(f"温度: {args.temperature}")
        print(f"Top-k: {args.top_k}")
        print("=" * 60 + "\n")
        
        for i in range(args.num_samples):
            print(f"样本 {i+1}/{args.num_samples}:")
            print("-" * 60)
            
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                args.length, args.temperature, args.top_k, device
            )
            
            print(generated_text)
            print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
