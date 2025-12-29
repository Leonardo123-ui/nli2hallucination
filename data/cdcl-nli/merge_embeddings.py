"""
合并分片的 embeddings 文件
"""

import torch
import glob
import os
import re
from pathlib import Path


def merge_embedding_files(input_dir, output_file):
    """
    合并分片的 embedding 文件

    Args:
        input_dir: 包含分片文件的目录
        output_file: 输出文件路径
    """
    print(f"正在合并 {input_dir} 中的 embeddings 文件...")

    # 查找所有 .npz 文件（排除目标文件本身）
    pattern = os.path.join(input_dir, "node_embeddings_*.npz")
    files = glob.glob(pattern)

    # 按数字顺序排序
    def extract_number(filename):
        match = re.search(r'node_embeddings_(\d+)\.npz', filename)
        return int(match.group(1)) if match else 0

    files = sorted(files, key=extract_number)

    if not files:
        print(f"❌ 未找到任何 .npz 文件: {pattern}")
        return False

    print(f"找到 {len(files)} 个分片文件:")
    for f in files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    # 加载并合并所有数据
    all_data = []
    total_samples = 0

    for idx, file_path in enumerate(files, 1):
        print(f"\n正在加载分片 {idx}/{len(files)}: {os.path.basename(file_path)}")
        try:
            data = torch.load(file_path)
            samples = len(data)
            all_data.extend(data)
            total_samples += samples
            print(f"  ✅ 加载了 {samples} 个样本")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            return False

    print(f"\n合并完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  第一个样本的 premise 节点数: {len(all_data[0]['premise'])}")
    print(f"  第一个样本的 hypothesis 节点数: {len(all_data[0]['hypothesis'])}")

    # 保存合并后的文件
    print(f"\n正在保存到: {output_file}")
    torch.save(all_data, output_file)

    output_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"✅ 保存成功! 文件大小: {output_size_mb:.1f} MB")

    # 可选：删除分片文件
    print("\n是否删除原始分片文件? (y/n): ", end='')
    # 自动选择不删除，以防万一
    print("n (保留原始文件)")

    return True


def merge_all_embeddings(base_dir):
    """
    合并训练集和测试集的 embeddings

    Args:
        base_dir: 数据基础目录
    """
    # 合并训练集
    train_dir = os.path.join(base_dir, "train")
    train_output = os.path.join(train_dir, "node_embeddings.npz")

    if not os.path.exists(train_output):
        print("="*60)
        print("合并训练集 embeddings")
        print("="*60)
        success = merge_embedding_files(train_dir, train_output)
        if success:
            print(f"\n✅ 训练集 embeddings 合并成功: {train_output}")
        else:
            print(f"\n❌ 训练集 embeddings 合并失败")
    else:
        print(f"✅ 训练集 embeddings 已存在: {train_output}")

    # 合并测试集
    test_dir = os.path.join(base_dir, "test")
    test_output = os.path.join(test_dir, "node_embeddings.npz")

    if os.path.exists(test_dir):
        test_files = glob.glob(os.path.join(test_dir, "node_embeddings_*.npz"))
        if test_files and not os.path.exists(test_output):
            print("\n" + "="*60)
            print("合并测试集 embeddings")
            print("="*60)
            success = merge_embedding_files(test_dir, test_output)
            if success:
                print(f"\n✅ 测试集 embeddings 合并成功: {test_output}")
            else:
                print(f"\n❌ 测试集 embeddings 合并失败")
        elif os.path.exists(test_output):
            print(f"\n✅ 测试集 embeddings 已存在: {test_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='合并分片的 embeddings 文件')
    parser.add_argument('--data_dir',
                       default='./data',
                       help='数据目录')
    parser.add_argument('--train_only',
                       action='store_true',
                       help='仅合并训练集')
    parser.add_argument('--test_only',
                       action='store_true',
                       help='仅合并测试集')

    args = parser.parse_args()

    if args.train_only:
        # 仅合并训练集
        train_dir = os.path.join(args.data_dir, "train")
        train_output = os.path.join(train_dir, "node_embeddings.npz")
        merge_embedding_files(train_dir, train_output)
    elif args.test_only:
        # 仅合并测试集
        test_dir = os.path.join(args.data_dir, "test")
        test_output = os.path.join(test_dir, "node_embeddings.npz")
        merge_embedding_files(test_dir, test_output)
    else:
        # 合并所有
        merge_all_embeddings(args.data_dir)

    print("\n" + "="*60)
    print("合并完成！可以继续运行 arrange_hallucination_data.py")
    print("="*60)
