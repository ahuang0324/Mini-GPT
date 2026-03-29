"""
数据准备脚本
用于下载和准备训练数据（唐诗或莎士比亚文本）

使用方法:
    python prepare_data.py --type tang_poetry --output ./data/tang_poetry.txt
    python prepare_data.py --type shakespeare --output ./data/shakespeare.txt
"""

import argparse
import os
import requests


def download_shakespeare(output_path):
    """
    下载莎士比亚作品集
    
    参数:
        output_path: 输出文件路径
    """
    print("下载莎士比亚作品集...")
    
    # 从 Project Gutenberg 下载
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        text = response.text
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ 成功下载 {len(text)} 个字符")
        print(f"✓ 保存到 {output_path}")
        
        # 显示预览
        print("\n文本预览:")
        print("-" * 60)
        print(text[:500])
        print("-" * 60)
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n备选方案：")
        print("1. 手动下载: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        print(f"2. 保存到: {output_path}")


def create_sample_tang_poetry(output_path):
    """
    创建示例唐诗数据
    
    参数:
        output_path: 输出文件路径
    """
    print("创建示例唐诗数据...")
    
    # 一些经典唐诗（示例）
    tang_poetry = """春晓
孟浩然
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

静夜思
李白
床前明月光，疑是地上霜。
举头望明月，低头思故乡。

登鹳雀楼
王之涣
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。

相思
王维
红豆生南国，春来发几枝。
愿君多采撷，此物最相思。

江雪
柳宗元
千山鸟飞绝，万径人踪灭。
孤舟蓑笠翁，独钓寒江雪。

寻隐者不遇
贾岛
松下问童子，言师采药去。
只在此山中，云深不知处。

春望
杜甫
国破山河在，城春草木深。
感时花溅泪，恨别鸟惊心。
烽火连三月，家书抵万金。
白头搔更短，浑欲不胜簪。

黄鹤楼送孟浩然之广陵
李白
故人西辞黄鹤楼，烟花三月下扬州。
孤帆远影碧空尽，唯见长江天际流。

枫桥夜泊
张继
月落乌啼霜满天，江枫渔火对愁眠。
姑苏城外寒山寺，夜半钟声到客船。

望庐山瀑布
李白
日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，疑是银河落九天。

早发白帝城
李白
朝辞白帝彩云间，千里江陵一日还。
两岸猿声啼不住，轻舟已过万重山。

赋得古原草送别
白居易
离离原上草，一岁一枯荣。
野火烧不尽，春风吹又生。
远芳侵古道，晴翠接荒城。
又送王孙去，萋萋满别情。

凉州词
王之涣
黄河远上白云间，一片孤城万仞山。
羌笛何须怨杨柳，春风不度玉门关。

出塞
王昌龄
秦时明月汉时关，万里长征人未还。
但使龙城飞将在，不教胡马度阴山。

芙蓉楼送辛渐
王昌龄
寒雨连江夜入吴，平明送客楚山孤。
洛阳亲友如相问，一片冰心在玉壶。

鹿柴
王维
空山不见人，但闻人语响。
返景入深林，复照青苔上。

竹里馆
王维
独坐幽篁里，弹琴复长啸。
深林人不知，明月来相照。

送元二使安西
王维
渭城朝雨浥轻尘，客舍青青柳色新。
劝君更尽一杯酒，西出阳关无故人。

九月九日忆山东兄弟
王维
独在异乡为异客，每逢佳节倍思亲。
遥知兄弟登高处，遍插茱萸少一人。

望月怀远
张九龄
海上生明月，天涯共此时。
情人怨遥夜，竟夕起相思。
灭烛怜光满，披衣觉露滋。
不堪盈手赠，还寝梦佳期。
"""
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tang_poetry)
    
    print(f"✓ 成功创建示例数据 {len(tang_poetry)} 个字符")
    print(f"✓ 保存到 {output_path}")
    
    print("\n注意: 这只是一个小型示例数据集。")
    print("为了获得更好的效果，建议使用更大的唐诗数据集。")
    print("\n可以从以下来源获取更多唐诗数据:")
    print("1. https://github.com/chinese-poetry/chinese-poetry")
    print("2. https://github.com/jackeyGao/chinese-poetry")
    
    # 显示预览
    print("\n文本预览:")
    print("-" * 60)
    print(tang_poetry[:300])
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--type', type=str, required=True,
                        choices=['tang_poetry', 'shakespeare'],
                        help='数据类型: tang_poetry 或 shakespeare')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.type == 'shakespeare':
        download_shakespeare(args.output)
    elif args.type == 'tang_poetry':
        create_sample_tang_poetry(args.output)


if __name__ == "__main__":
    main()
