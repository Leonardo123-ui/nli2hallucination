#!/usr/bin/env python3
"""
解析 Qwen 批量推理结果文件
计算幻觉检测性能指标
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_llm_prediction(llm_text: str) -> bool:
    """
    从LLM输出提取幻觉判断

    标准格式中，【验证】部分包含：
    - "是/存在幻觉" → True
    - "否/不存在幻觉" → False
    - "部分准确" → False（保守处理）
    """
    if not llm_text:
        return False

    text = llm_text.strip()

    # 优先级1：寻找【验证】标签的明确判断
    verification_start = text.find('【验证】')
    if verification_start == -1:
        verification_start = text.find('验证】')

    if verification_start != -1:
        # 取【验证】后面的内容（到下一个【或结尾）
        verification_end = text.find('【', verification_start + 2)
        if verification_end == -1:
            verification_end = len(text)

        verification_section = text[verification_start:verification_end].lower()

        # 检查明确的判断
        if '是' in verification_section or '存在幻觉' in verification_section or '有幻觉' in verification_section:
            if '否' not in verification_section[:100]:
                return True

        if '否' in verification_section or '不存在幻觉' in verification_section or '无幻觉' in verification_section:
            return False

        if '部分准确' in verification_section:
            return False

    # 降级：全文检查
    text_lower = text.lower()

    if '是' in text_lower and '否' not in text_lower[:500]:
        return True

    return False


def parse_batch_result(result_file: str, cache_file: str) -> Tuple[List[int], List[int], Dict]:
    """
    解析批量推理结果

    Returns:
        (true_labels, predictions, results_dict)
    """
    logger.info(f"\n加载文件: {result_file}")

    # 加载缓存
    if not Path(cache_file).exists():
        logger.error(f"缓存文件不存在: {cache_file}")
        return [], [], {}

    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    logger.info(f"✓ 加载缓存: {len(cache)} 个样本")

    # 加载推理结果
    results = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                custom_id = item.get('custom_id')

                # 解析响应
                response_body = item.get('response', {}).get('body', {})
                if not response_body:
                    logger.warning(f"行 {line_num}: 无响应体")
                    continue

                choices = response_body.get('choices', [])
                if not choices:
                    logger.warning(f"行 {line_num}: 无 choices")
                    continue

                message = choices[0].get('message', {})
                llm_text = message.get('content', '')

                results[custom_id] = {
                    'llm_text': llm_text,
                    'pred': extract_llm_prediction(llm_text)
                }

            except Exception as e:
                logger.warning(f"行 {line_num} 解析失败: {str(e)[:100]}")

    logger.info(f"✓ 解析了 {len(results)} 条推理结果")

    # 对齐标签
    true_labels = []
    predictions = []
    detail_results = []
    matched = 0
    missing = 0

    for custom_id, result in sorted(results.items(), key=lambda x: int(x[0])):
        # 从缓存获取真实标签
        if custom_id not in cache:
            logger.warning(f"样本 {custom_id} 不在缓存中，跳过")
            missing += 1
            continue

        cache_data = cache[custom_id]
        if isinstance(cache_data, dict):
            true_label = cache_data['rst_pred']  # 使用 RST-RGAT 的全局判别作为真实标签
        else:
            true_label = cache_data[1]  # 旧格式

        true_labels.append(true_label)
        predictions.append(1 if result['pred'] else 0)
        matched += 1

        detail_results.append({
            'sample_id': custom_id,
            'true_label': true_label,
            'pred': 1 if result['pred'] else 0,
            'llm_text': result['llm_text'][:200]
        })

    logger.info(f"✓ 对齐结果: {matched} 个匹配，{missing} 个缺失")

    return true_labels, predictions, {
        'details': detail_results,
        'matched': matched,
        'missing': missing
    }


def calculate_metrics(true_labels: List[int], predictions: List[int]) -> Dict:
    """计算性能指标"""

    if len(true_labels) == 0:
        logger.error("没有有效的标签")
        return {}

    # 二分类指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()

    # Macro F1（多分类习惯）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_samples': len(true_labels),
        'positive_ratio': sum(true_labels) / len(true_labels) if true_labels else 0
    }


def main():
    # 配置
    BASE_DIR = Path("/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec")
    RESULT_FILE = BASE_DIR / "batch_inference_result" / "qwen-plus-type1-result.jsonl"
    CACHE_FILE = BASE_DIR / "test_evaluation" / "rst_cache.pkl"

    print("="*80)
    print("解析批量推理结果并计算指标")
    print("="*80)

    # 解析结果
    true_labels, predictions, details = parse_batch_result(str(RESULT_FILE), str(CACHE_FILE))

    if not true_labels:
        logger.error("无法解析结果")
        return

    # 计算指标
    metrics = calculate_metrics(true_labels, predictions)

    # 打印结果
    print("\n" + "="*80)
    print("性能指标")
    print("="*80)
    print(f"\n样本统计:")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  正样本比例: {metrics['positive_ratio']:.1%}")
    print(f"  匹配数: {details['matched']}")
    print(f"  缺失数: {details['missing']}")

    print(f"\n【主要指标】")
    print(f"  F1 (Binary): {metrics['f1']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")

    print(f"\n【Macro 指标（与训练对齐）】")
    print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"  Precision:  {metrics['precision_macro']:.4f}")
    print(f"  Recall:     {metrics['recall_macro']:.4f}")

    print(f"\n【混淆矩阵】")
    print(f"  TP (正确检测幻觉): {metrics['tp']}")
    print(f"  TN (正确未检测): {metrics['tn']}")
    print(f"  FP (误报): {metrics['fp']}")
    print(f"  FN (漏报): {metrics['fn']}")

    print(f"\n【样本示例】")
    for result in details['details'][:5]:
        status = "✓" if result['pred'] == result['true_label'] else "✗"
        true_text = "有幻觉" if result['true_label'] == 1 else "无幻觉"
        pred_text = "有幻觉" if result['pred'] == 1 else "无幻觉"
        print(f"\n  {status} 样本 {result['sample_id']}")
        print(f"    真实: {true_text}, 预测: {pred_text}")
        print(f"    LLM: {result['llm_text'][:80]}...")

    # 保存详细结果
    output_file = BASE_DIR / "test_evaluation" / "qwen-plus-type1-metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'sample_count': details['matched'],
            'file': str(RESULT_FILE)
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 详细结果已保存到: {output_file}")

    # 统计错误分布
    print("\n" + "="*80)
    print("错误分析")
    print("="*80)

    false_positives = [d for d in details['details'] if d['pred'] == 1 and d['true_label'] == 0]
    false_negatives = [d for d in details['details'] if d['pred'] == 0 and d['true_label'] == 1]

    print(f"\n误报（FP）: {len(false_positives)} 个")
    if false_positives:
        print("  前 3 个误报样本:")
        for fp in false_positives[:3]:
            print(f"    - 样本 {fp['sample_id']}: {fp['llm_text'][:60]}...")

    print(f"\n漏报（FN）: {len(false_negatives)} 个")
    if false_negatives:
        print("  前 3 个漏报样本:")
        for fn in false_negatives[:3]:
            print(f"    - 样本 {fn['sample_id']}: {fn['llm_text'][:60]}...")

if __name__ == "__main__":
    main()
