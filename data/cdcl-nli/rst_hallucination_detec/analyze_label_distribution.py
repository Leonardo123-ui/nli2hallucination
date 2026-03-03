#!/usr/bin/env python3
"""
展示训练集和测试集的标签分布统计
"""

import sys
sys.path.insert(0, "/mnt/nlp/yuanmengying/CDCL_NLI-old")

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from data_preprocessing import split_by_dataset_field


def analyze_label_distribution():
    """分析标签分布"""

    # 加载已处理的数据
    processed_data_path = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec/outputs/processed_data.json"

    logger.info("加载已处理的数据...")
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    logger.info(f"✓ 加载完成，共 {len(processed_data)} 条样本")

    # 划分数据集
    train_data, test_data = split_by_dataset_field(processed_data)

    # 统计训练集
    train_global_labels = [sample['y_global'] for sample in train_data]
    train_pos_count = sum(train_global_labels)
    train_neg_count = len(train_global_labels) - train_pos_count
    train_total = len(train_global_labels)
    train_pos_ratio = 100 * train_pos_count / train_total if train_total > 0 else 0

    logger.info(f"\n【训练集标签分布】")
    logger.info(f"  总样本: {train_total}")
    logger.info(f"  正例（有幻觉）: {train_pos_count} ({train_pos_ratio:.2f}%)")
    logger.info(f"  负例（无幻觉）: {train_neg_count} ({100-train_pos_ratio:.2f}%)")
    logger.info(f"  样本比例: {train_pos_count}:{train_neg_count}")

    # 统计测试集
    test_global_labels = [sample['y_global'] for sample in test_data]
    test_pos_count = sum(test_global_labels)
    test_neg_count = len(test_global_labels) - test_pos_count
    test_total = len(test_global_labels)
    test_pos_ratio = 100 * test_pos_count / test_total if test_total > 0 else 0

    logger.info(f"\n【测试集标签分布】")
    logger.info(f"  总样本: {test_total}")
    logger.info(f"  正例（有幻觉）: {test_pos_count} ({test_pos_ratio:.2f}%)")
    logger.info(f"  负例（无幻觉）: {test_neg_count} ({100-test_pos_ratio:.2f}%)")
    logger.info(f"  样本比例: {test_pos_count}:{test_neg_count}")

    # 总体统计
    total_samples = train_total + test_total
    total_pos = train_pos_count + test_pos_count
    total_neg = train_neg_count + test_neg_count

    logger.info(f"\n【总体统计】")
    logger.info(f"  总样本: {total_samples}")
    logger.info(f"  正例（有幻觉）: {total_pos} ({100*total_pos/total_samples:.2f}%)")
    logger.info(f"  负例（无幻觉）: {total_neg} ({100*total_neg/total_samples:.2f}%)")
    logger.info(f"  样本比例: {total_pos}:{total_neg}")

    # 对比分析
    logger.info(f"\n【数据集特征】")
    logger.info(f"  训练-测试比例: {train_total}:{test_total}")
    logger.info(f"  训练集不平衡度: {train_pos_ratio:.2f}% (正例)")
    logger.info(f"  测试集不平衡度: {test_pos_ratio:.2f}% (正例)")

    # 教育统计
    logger.info(f"\n【EDU级标签分布（训练集）】")
    train_edu_labels = []
    for sample in train_data:
        train_edu_labels.extend(sample['output_edu_labels'])

    if train_edu_labels:
        edu_pos = sum(train_edu_labels)
        edu_neg = len(train_edu_labels) - edu_pos
        edu_pos_ratio = 100 * edu_pos / len(train_edu_labels)
        logger.info(f"  总EDU数: {len(train_edu_labels)}")
        logger.info(f"  幻觉EDU: {edu_pos} ({edu_pos_ratio:.2f}%)")
        logger.info(f"  正常EDU: {edu_neg} ({100-edu_pos_ratio:.2f}%)")

    logger.info("\n✓ 标签分析完成\n")

    # 返回统计结果供后续使用
    return {
        'train': {
            'total': train_total,
            'positive': train_pos_count,
            'negative': train_neg_count,
            'positive_ratio': train_pos_ratio
        },
        'test': {
            'total': test_total,
            'positive': test_pos_count,
            'negative': test_neg_count,
            'positive_ratio': test_pos_ratio
        }
    }


if __name__ == "__main__":
    try:
        stats = analyze_label_distribution()
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
