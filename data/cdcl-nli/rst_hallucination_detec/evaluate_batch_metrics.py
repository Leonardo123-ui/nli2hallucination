#!/usr/bin/env python3
"""
批量推理结果解析和指标计算脚本
功能：
1. 严格的LLM提取逻辑（按【验证】部分的明确判断）
2. 单独的LLM、RST-RGAT指标
3. OR、AND、加权投票等ensemble指标
4. 详细的性能对比
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_llm_prediction(llm_text: str) -> bool:
    """
    提取LLM预测 - 支持多种输出格式

    优先级：
    1. 【结论】（Type 3.1新格式 - 双盲核查法）
    2. 【最终结论】（Type 1/2格式）
    3. 【验证】（旧格式）
    """
    if not llm_text:
        return False

    text = llm_text.strip()

    # 优先级1：查找【结论】标签（Type 3.1新格式）
    conclusion_start = text.find('【结论】')
    if conclusion_start != -1:
        conclusion_section = text[conclusion_start:conclusion_start+20]
        if '1' in conclusion_section:
            return True
        elif '0' in conclusion_section:
            return False

    # 优先级2：查找【最终结论】标签（Type 1/2格式）
    final_conclusion_start = text.find('【最终结论】')
    if final_conclusion_start != -1:
        conclusion_section = text[final_conclusion_start:final_conclusion_start+20]
        if '1' in conclusion_section:
            return True
        elif '0' in conclusion_section:
            return False

    # 降级：【验证】标签（旧格式）
    verification_start = text.find('【验证】')
    if verification_start == -1:
        verification_start = text.find('验证】')

    if verification_start != -1:
        verification_end = text.find('【', verification_start + 2)
        if verification_end == -1:
            verification_end = len(text)

        verification_section = text[verification_start:verification_end].lower()

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


def parse_batch_result(result_file: str, cache_file: str) -> Tuple[List[int], List[int], List[int], List[float], Dict]:
    """
    解析批量推理结果

    Args:
        result_file: JSONL格式的推理结果文件
        cache_file: 缓存文件（包含RST-RGAT标签）

    Returns:
        (true_labels, rst_predictions, llm_predictions, rst_probs, results_dict)
    """
    logger.info(f"\n加载文件: {result_file}")

    # 加载缓存
    if not Path(cache_file).exists():
        logger.error(f"缓存文件不存在: {cache_file}")
        return [], [], [], {}

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
    rst_predictions = []
    rst_probs = []
    llm_predictions = []
    detail_results = []
    matched = 0
    missing = 0

    for custom_id, result in sorted(results.items(), key=lambda x: int(x[0])):
        # 从缓存获取真实标签和RST预测
        if custom_id not in cache:
            logger.warning(f"样本 {custom_id} 不在缓存中，跳过")
            missing += 1
            continue

        cache_data = cache[custom_id]
        if isinstance(cache_data, dict):
            true_label = cache_data['true_label']  # 使用真实的标签
            rst_pred = cache_data['rst_pred']     # RST预测结果
            rst_prob = cache_data.get('rst_prob', 0.5)
        else:
            # 旧格式：(true_label, rst_pred, rst_prob)
            true_label = cache_data[0]
            rst_pred = cache_data[1]
            rst_prob = cache_data[2] if len(cache_data) > 2 else 0.5

        true_labels.append(true_label)
        rst_predictions.append(rst_pred)
        rst_probs.append(rst_prob)
        llm_predictions.append(1 if result['pred'] else 0)
        matched += 1

        detail_results.append({
            'sample_id': custom_id,
            'true_label': true_label,
            'rst_pred': rst_pred,
            'rst_prob': rst_prob,
            'llm_pred': 1 if result['pred'] else 0,
            'llm_text': result['llm_text'][:200]
        })

    logger.info(f"✓ 对齐结果: {matched} 个匹配，{missing} 个缺失")

    return true_labels, rst_predictions, llm_predictions, rst_probs, {
        'details': detail_results,
        'matched': matched,
        'missing': missing
    }


def calculate_metrics(true_labels: List[int], predictions: List[int], prefix: str = "") -> Dict:
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


def calculate_ensemble_metrics(
    true_labels: List[int],
    rst_preds: List[int],
    llm_preds: List[int],
    ensemble_type: str = 'or'
) -> Dict:
    """
    计算ensemble后的指标

    Args:
        true_labels: 真实标签
        rst_preds: RST-RGAT预测
        llm_preds: LLM预测
        ensemble_type: 集成方法 ('or', 'and', 'weighted')

    Returns:
        指标字典
    """
    if ensemble_type == 'or':
        # OR操作：只要有一个预测1，就预测1
        ensemble_preds = [1 if (r == 1 or l == 1) else 0 for r, l in zip(rst_preds, llm_preds)]
    elif ensemble_type == 'and':
        # AND操作：两个都预测1，才预测1
        ensemble_preds = [1 if (r == 1 and l == 1) else 0 for r, l in zip(rst_preds, llm_preds)]
    elif ensemble_type == 'weighted':
        # 加权投票：RST权重0.4，LLM权重0.6
        ensemble_preds = [1 if (0.4 * r + 0.6 * l >= 0.5) else 0 for r, l in zip(rst_preds, llm_preds)]
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return calculate_metrics(true_labels, ensemble_preds)


def calculate_probability_fusion_ensemble(
    true_labels: List[int],
    rst_preds: List[int],
    rst_probs: List[float],
    llm_preds: List[int],
    rst_weight: float = 0.6
) -> Dict:
    """
    计算概率级别的ensemble指标 (Scheme 4)

    特点：在概率级别融合，而非决策级别
    - 使用RST的实际概率值（更可靠）
    - LLM预测转化为高置信度概率（0.95/0.05）
    - 加权平均后threshold=0.5

    Args:
        true_labels: 真实标签
        rst_preds: RST-RGAT预测（用于备选）
        rst_probs: RST-RGAT概率
        llm_preds: LLM预测
        rst_weight: RST权重（LLM权重=1-rst_weight）

    Returns:
        指标字典
    """
    llm_weight = 1 - rst_weight
    ensemble_preds = []

    for rst_prob, llm_pred in zip(rst_probs, llm_preds):
        # LLM预测转化为高置信度概率
        llm_prob = 0.95 if llm_pred == 1 else 0.05
        # 概率级融合
        fused_prob = rst_weight * rst_prob + llm_weight * llm_prob
        # Threshold at 0.5
        ensemble_preds.append(1 if fused_prob >= 0.5 else 0)

    return calculate_metrics(true_labels, ensemble_preds)

    """打印指标摘要"""
    print(f"\n{title}")
    print(f"{prefix}F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"{prefix}F1 (Binary): {metrics['f1']:.4f}")
    print(f"{prefix}Precision:  {metrics['precision']:.4f}")
    print(f"{prefix}Recall:     {metrics['recall']:.4f}")
    print(f"{prefix}Accuracy:   {metrics['accuracy']:.4f}")


def print_detailed_metrics(metrics: Dict, title: str):
    """打印详细指标"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"\n样本统计:")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  正样本比例: {metrics['positive_ratio']:.1%}")

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


def print_sample_analysis(details: List[Dict], title: str, num_samples: int = 5):
    """打印样本分析"""
    print(f"\n{title}")
    for result in details[:num_samples]:
        status = "✓" if result.get('llm_pred') == result['true_label'] else "✗"
        true_text = "有幻觉" if result['true_label'] == 1 else "无幻觉"
        pred_text = "有幻觉" if result.get('llm_pred') == 1 else "无幻觉"
        print(f"\n  {status} 样本 {result['sample_id']}")
        print(f"    真实: {true_text}, LLM预测: {pred_text}")
        print(f"    LLM: {result['llm_text']}...")


def main():
    """处理所有JSONL批量推理结果，整合指标"""
    BASE_DIR = Path("/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec")
    result_file_dir = BASE_DIR / "batch_inference_result_version2"
    OUTPUT_DIR = BASE_DIR / "test_evaluation"

    # 获取缓存文件
    CACHE_FILE = BASE_DIR / "test_evaluation" / "rst_cache_noedus.pkl"
    if not Path(CACHE_FILE).exists():
        CACHE_FILE = BASE_DIR / "test_evaluation" / "rst_cache_old.pkl"

    # 获取所有JSONL文件
    jsonl_files = list(result_file_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"在 {result_file_dir} 中未找到 JSONL 文件")
        return

    print("=" * 100)
    print("批量推理结果解析和指标计算（严格提取，包含RST-RGAT耦合）")
    print("=" * 100)
    print(f"\n找到 {len(jsonl_files)} 个结果文件\n")

    # 存储所有指标
    all_results = {}

    # 处理每个文件
    for result_file in jsonl_files:
        file_name = result_file.stem.replace("-result", "")
        print(f"处理: {file_name}...", end=" ", flush=True)

        # 解析结果
        true_labels, rst_preds, llm_preds, rst_probs, details = parse_batch_result(
            str(result_file), str(CACHE_FILE)
        )

        if not true_labels:
            logger.error(f"✗ 解析失败")
            continue

        # 计算指标
        llm_metrics = calculate_metrics(true_labels, llm_preds)
        rst_metrics = calculate_metrics(true_labels, rst_preds)
        or_metrics = calculate_ensemble_metrics(true_labels, rst_preds, llm_preds, 'or')
        and_metrics = calculate_ensemble_metrics(true_labels, rst_preds, llm_preds, 'and')
        weighted_metrics = calculate_ensemble_metrics(true_labels, rst_preds, llm_preds, 'weighted')
        prob_fusion_metrics = calculate_probability_fusion_ensemble(true_labels, rst_preds, rst_probs, llm_preds, rst_weight=0.6)

        # 存储结果
        all_results[file_name] = {
            'llm': llm_metrics,
            'rst': rst_metrics,
            'or_ensemble': or_metrics,
            'and_ensemble': and_metrics,
            'weighted_ensemble': weighted_metrics,
            'prob_fusion_ensemble': prob_fusion_metrics,
            'samples': details['details']
        }

        print(f"✓ (F1_Macro: LLM={llm_metrics['f1_macro']:.4f}, RST={rst_metrics['f1_macro']:.4f}, ProbFusion={prob_fusion_metrics['f1_macro']:.4f})")

    if not all_results:
        logger.error("没有成功处理任何文件")
        return

    # 打印汇总对比表
    print("\n" + "=" * 160)
    print("性能对比汇总表")
    print("=" * 160)

    print(f"\n{'文件名':<40} {'LLM F1M':<12} {'RST F1M':<12} {'OR F1M':<12} {'AND F1M':<12} {'Weight F1M':<12} {'ProbFus F1M':<12}")
    print("-" * 160)

    for file_name in sorted(all_results.keys()):
        results = all_results[file_name]
        print(f"{file_name:<40} "
              f"{results['llm']['f1_macro']:.4f}      "
              f"{results['rst']['f1_macro']:.4f}      "
              f"{results['or_ensemble']['f1_macro']:.4f}      "
              f"{results['and_ensemble']['f1_macro']:.4f}      "
              f"{results['weighted_ensemble']['f1_macro']:.4f}      "
              f"{results['prob_fusion_ensemble']['f1_macro']:.4f}")

    # 详细对比表
    print("\n\n" + "=" * 150)
    print("详细指标对比")
    print("=" * 150)

    for file_name in sorted(all_results.keys()):
        results = all_results[file_name]
        print(f"\n【{file_name}】")
        print(f"{'方法':<20} {'F1 Macro':<12} {'F1 Binary':<12} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
        print("-" * 80)

        methods = [
            ("LLM", results['llm']),
            ("RST-RGAT", results['rst']),
            ("OR Ensemble", results['or_ensemble']),
            ("AND Ensemble", results['and_ensemble']),
            ("Weighted", results['weighted_ensemble']),
            ("ProbFusion", results['prob_fusion_ensemble'])
        ]

        for method_name, metrics in methods:
            print(f"{method_name:<20} "
                  f"{metrics['f1_macro']:.4f}       "
                  f"{metrics['f1']:.4f}       "
                  f"{metrics['precision']:.4f}       "
                  f"{metrics['recall']:.4f}       "
                  f"{metrics['accuracy']:.4f}")

    # 保存完整结果到JSON
    output_file = OUTPUT_DIR / "batch-metrics-all-results- version2.json"

    # 转换为可序列化格式
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, float) else int(obj)
        else:
            return obj

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_types(all_results), f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完整结果已保存到: {output_file}")

    # 生成性能提升分析
    print("\n" + "=" * 100)
    print("性能提升分析")
    print("=" * 100)

    # 按类型分组（model-typeX）
    type_groups = {}
    for file_name in sorted(all_results.keys()):
        # 解析: model-typeX 格式
        parts = file_name.rsplit("-type", 1)
        if len(parts) == 2:
            model = parts[0]
            prompt_type = f"Type{parts[1]}"

            if model not in type_groups:
                type_groups[model] = {}
            type_groups[model][prompt_type] = all_results[file_name]

    # 对比各Type的性能
    for model in sorted(type_groups.keys()):
        types = type_groups[model]
        if len(types) > 1:
            print(f"\n【{model} - Prompt类型对比】")

            for type_name in sorted(types.keys()):
                llm_f1 = types[type_name]['llm']['f1_macro']
                print(f"  {type_name}: LLM F1_Macro = {llm_f1:.4f}")

            # 计算提升
            type_names = sorted(types.keys())
            if len(type_names) >= 2:
                type1_f1 = types[type_names[0]]['llm']['f1_macro']
                for type_name in type_names[1:]:
                    current_f1 = types[type_name]['llm']['f1_macro']
                    improvement = (current_f1 - type1_f1) / type1_f1 * 100
                    print(f"    {type_name} vs {type_names[0]}: {improvement:+.1f}%")

    print("\n✓ 分析完成")


if __name__ == "__main__":
    main()
