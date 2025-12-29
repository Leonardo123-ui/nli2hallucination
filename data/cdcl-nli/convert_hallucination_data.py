"""
将幻觉检测数据转换为 CDCL-NLI 所需的格式

输入：summary_data.json
输出：JSON格式的训练集和测试集，适配CDCL-NLI的数据格式
"""

import json
import os
from pathlib import Path


def convert_hallucination_to_nli_format(json_path, output_dir):
    """
    将幻觉检测数据转换为 NLI 格式

    Args:
        json_path: JSON数据文件路径
        output_dir: 输出目录

    Returns:
        train_path, test_path: 训练集和测试集的路径
    """
    print(f"正在加载数据: {json_path}")

    # 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"数据总量: {len(data)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据dataset字段分割训练集和测试集
    train_data_raw = [item for item in data if item.get('dataset') == 'train']
    test_data_raw = [item for item in data if item.get('dataset') == 'test']

    print(f"训练集: {len(train_data_raw)}")
    print(f"测试集: {len(test_data_raw)}")

    # 转换训练集
    train_data = convert_to_nli_format(train_data_raw)
    train_path = os.path.join(output_dir, 'hallucination_train.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存: {train_path}")

    # 转换测试集
    test_data = convert_to_nli_format(test_data_raw)
    test_path = os.path.join(output_dir, 'hallucination_test.json')

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存: {test_path}")

    # 打印统计信息
    print_statistics(train_data, test_data)

    return train_path, test_path


def convert_to_nli_format(data_list):
    """
    将数据列表转换为CDCL-NLI格式

    CDCL-NLI格式：
    - news1_origin: 前提（context）
    - news2_origin: 假设（output）
    - label: 标签（根据hallucination_labels_processed判断）
    """
    nli_data = []

    for item in data_list:
        # 获取幻觉标签
        hallucination_labels = item.get('hallucination_labels_processed', {})

        # 判断是否有幻觉：
        # evident_conflict=1 或 baseless_info=1 表示有幻觉
        has_hallucination = (
            hallucination_labels.get('evident_conflict', 0) > 0 or
            hallucination_labels.get('baseless_info', 0) > 0
        )

        # 转换为二分类标签
        original_label = 1 if has_hallucination else 0

        nli_item = {
            "news1_origin": item['context'],      # context作为前提
            "news2_origin": item['output'],       # output作为假设
            "label": map_label(original_label),   # 转换标签
            "original_label": original_label,     # 保留原始标签
            "id": str(item['id']),
            "task_type": item.get('task_type', 'Summary'),
            "quality": item.get('quality', 'unknown'),
            "model": item.get('model', 'unknown'),
            "hallucination_info": hallucination_labels
        }
        nli_data.append(nli_item)

    return nli_data


def map_label(hallucination_label):
    """
    将幻觉检测标签映射到NLI标签

    Args:
        hallucination_label: 0=无幻觉, 1=有幻觉

    Returns:
        NLI标签：0=entailment（无幻觉）, 2=contradiction（有幻觉）

    说明：
    - 无幻觉（0）：output与context一致 -> entailment (0)
    - 有幻觉（1）：output与context矛盾 -> contradiction (2)
    """
    if hallucination_label == 0:
        return 0  # entailment
    elif hallucination_label == 1:
        return 2  # contradiction
    else:
        raise ValueError(f"未知的标签: {hallucination_label}")


def print_statistics(train_data, test_data):
    """打印数据统计信息"""
    print("\n" + "="*60)
    print("数据转换统计")
    print("="*60)

    print(f"\n训练集:")
    print(f"  总样本数: {len(train_data)}")
    train_labels = [item['original_label'] for item in train_data]
    print(f"  无幻觉 (0): {train_labels.count(0)}")
    print(f"  有幻觉 (1): {train_labels.count(1)}")

    print(f"\n测试集:")
    print(f"  总样本数: {len(test_data)}")
    test_labels = [item['original_label'] for item in test_data]
    print(f"  无幻觉 (0): {test_labels.count(0)}")
    print(f"  有幻觉 (1): {test_labels.count(1)}")

    print("\nNLI标签映射:")
    print("  0 (无幻觉) -> 0 (entailment)")
    print("  1 (有幻觉) -> 2 (contradiction)")

    print("\n幻觉判断规则:")
    print("  evident_conflict > 0 或 baseless_info > 0 -> 有幻觉")
    print("="*60 + "\n")


def create_small_sample(json_path, sample_size=100, output_suffix='_sample'):
    """
    创建小样本数据用于快速测试

    Args:
        json_path: JSON数据文件路径
        sample_size: 样本大小
        output_suffix: 输出文件后缀
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机抽样
    import random
    random.seed(42)
    sample_data = random.sample(data, min(sample_size, len(data)))

    # 保存样本
    sample_path = json_path.replace('.json', f'{output_suffix}.json')
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"小样本数据已保存: {sample_path} ({len(sample_data)} 样本)")
    return sample_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='转换幻觉检测数据为CDCL-NLI格式')
    parser.add_argument('--json_path',
                       default='/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json',
                       help='JSON数据文件路径')
    parser.add_argument('--output_dir',
                       default='./data',
                       help='输出目录')
    parser.add_argument('--create_sample',
                       action='store_true',
                       help='是否创建小样本数据')
    parser.add_argument('--sample_size',
                       type=int,
                       default=100,
                       help='小样本大小')

    args = parser.parse_args()

    # 转换数据
    train_path, test_path = convert_hallucination_to_nli_format(
        args.json_path,
        args.output_dir
    )

    # 创建小样本
    if args.create_sample:
        print("\n创建小样本数据用于测试...")
        create_small_sample(train_path, args.sample_size)
        create_small_sample(test_path, min(50, args.sample_size))
