#!/usr/bin/env python3
"""
使用 vLLM 加速 LLM 评估
vLLM 支持 Continuous Batching，可以实现 3-5x 加速，无需担心并发数值问题
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, "/mnt/nlp/yuanmengying/CDCL_NLI-old")

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM 未安装，请运行: pip install vllm")


class VLLMEvaluator:
    """基于 vLLM 的高速评估器"""

    def __init__(self, qwen_path: str):
        """初始化 vLLM"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM 未安装，请先运行: pip install vllm")

        logger.info(f"初始化 vLLM: {qwen_path}")
        self.llm = LLM(
            model=qwen_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.65,  # 进一步降低显存占用
            max_model_len=2048,
            max_num_seqs=8  # 限制并发请求数，降低显存占用
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            top_k=50
        )
        logger.info("✓ vLLM 初始化完成")

    def _build_prompt(self, context: str, output: str, detection_result: Dict) -> str:
        """构建单个prompt"""
        global_hallucination = detection_result['global_hallucination']
        global_prob = detection_result['global_prob']

        prompt = f"""你是一个专业的幻觉检测分析师。

【待分析文本】
原始文章：{context}...

生成摘要：{output}...

【自动检测结果】
RST-RGAT模型检测：{"发现幻觉" if global_hallucination else "未发现幻觉"}（置信度：{global_prob:.1%}）

请完成以下任务：
1. 【验证】判断自动检测是否准确（是/否/部分准确），并说明理由
2. 【定位】指出摘要中具体哪些内容与原文不符
3. 【分析】解释产生幻觉的原因（事实冲突 / 无中生有 / 过度推断等）
4. 【建议】给出修正建议"""

        return prompt

    def analyze_batch(
        self,
        contexts: List[str],
        outputs: List[str],
        detection_results: List[Dict],
    ) -> List[str]:
        """
        使用 vLLM 批处理分析

        Args:
            contexts: 原始文本列表
            outputs: 生成文本列表
            detection_results: 检测结果列表

        Returns:
            分析文本列表
        """
        prompts = [
            self._build_prompt(context, output, result)
            for context, output, result in zip(contexts, outputs, detection_results)
        ]

        # vLLM 的 continuous batching 会自动处理批处理
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = [output.outputs[0].text for output in outputs]
        return results


def extract_llm_prediction(llm_text: str) -> bool:
    """保守的LLM判断提取函数"""
    if not llm_text:
        return False

    # 优先级1：【验证】标签明确判断
    if '【验证】是' in llm_text or '验证】是' in llm_text:
        return True
    if '【验证】否' in llm_text or '验证】否' in llm_text:
        return False

    # 【验证】部分准确：保守处理
    if '【验证】部分准确' in llm_text or '验证】部分准确' in llm_text:
        return False
    if '部分准确' in llm_text[:1000]:
        return False

    # 启发式规则
    text_head = llm_text[:500].lower()
    if '是' in text_head and '否' not in text_head[:200]:
        return True

    return False


def evaluate_with_vllm(
    test_data: List[Dict],
    rst_cache: Dict,
    qwen_path: str,
    batch_size: int = 16
) -> Dict:
    """
    使用 vLLM 进行高速评估

    Args:
        test_data: 测试数据
        rst_cache: RST 结果缓存
        qwen_path: Qwen 模型路径
        batch_size: vLLM 批处理大小

    Returns:
        评估结果
    """
    logger.info(f"初始化 vLLM 评估器 (batch_size={batch_size})...")
    evaluator = VLLMEvaluator(qwen_path)

    logger.info(f"开始评估 {len(test_data)} 个样本...")

    true_labels = []
    rst_preds = []
    rst_probs = []
    llm_preds = []
    llm_probs = []
    detailed_results = []

    # 收集待处理的样本
    batch_contexts = []
    batch_outputs = []
    batch_results = []
    batch_indices = []
    batch_cache_info = []

    with tqdm(total=len(test_data), desc="vLLM 评估进度") as pbar:
        for idx, sample in enumerate(test_data):
            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            # 处理真实标签
            hall_labels = sample.get('hallucination_labels')
            if isinstance(hall_labels, str):
                try:
                    hall_labels = json.loads(hall_labels)
                except (json.JSONDecodeError, TypeError):
                    hall_labels = []
            true_label = 1 if hall_labels else 0

            # 从缓存获取 RST 结果
            if sample_id not in rst_cache:
                logger.warning(f"样本 {sample_id} 不在缓存中，跳过")
                pbar.update(1)
                continue

            true_label_cached, rst_pred, rst_prob = rst_cache[sample_id]

            # 构建 mock detection_result
            mock_detection_result = {
                'global_hallucination': bool(rst_pred),
                'global_prob': float(rst_prob),
                'hallucination_edus': []
            }

            # 收集到批处理缓冲区
            batch_contexts.append(context)
            batch_outputs.append(output)
            batch_results.append(mock_detection_result)
            batch_indices.append(idx)
            batch_cache_info.append((sample_id, true_label, rst_pred, rst_prob))

            # 当达到 batch_size 时，进行批处理
            if len(batch_contexts) >= batch_size or idx == len(test_data) - 1:
                try:
                    # 使用 vLLM 批处理
                    llm_analyses = evaluator.analyze_batch(
                        batch_contexts, batch_outputs, batch_results
                    )

                    # 处理结果
                    for llm_analysis, (sample_id, true_label, rst_pred, rst_prob) in zip(
                        llm_analyses, batch_cache_info
                    ):
                        true_labels.append(true_label)
                        rst_preds.append(rst_pred)
                        rst_probs.append(rst_prob)

                        llm_pred = 1 if extract_llm_prediction(llm_analysis) else 0
                        llm_preds.append(llm_pred)
                        llm_probs.append(0.6 if llm_pred == 1 else 0.4)

                        detail = {
                            'id': sample_id,
                            'true_label': true_label,
                            'rst_prediction': rst_pred,
                            'rst_prob': float(rst_prob),
                            'llm_prediction': llm_pred,
                            'llm_prob': float(0.6 if llm_pred == 1 else 0.4),
                            'llm_analysis': llm_analysis[:500],
                            'correct_rst': rst_pred == true_label,
                            'correct_llm': llm_pred == true_label
                        }
                        detailed_results.append(detail)

                    # 清空缓冲区
                    batch_contexts = []
                    batch_outputs = []
                    batch_results = []
                    batch_indices = []
                    batch_cache_info = []

                except Exception as e:
                    logger.error(f"批处理失败: {e}")
                    # 添加占位符
                    for sample_id, true_label, rst_pred, rst_prob in batch_cache_info:
                        true_labels.append(true_label)
                        rst_preds.append(rst_pred)
                        rst_probs.append(rst_prob)
                        llm_preds.append(0)
                        llm_probs.append(0.5)
                        detailed_results.append({
                            'id': sample_id,
                            'error': str(e),
                            'true_label': true_label
                        })

                    batch_contexts = []
                    batch_outputs = []
                    batch_results = []
                    batch_indices = []
                    batch_cache_info = []

            pbar.update(1)

    logger.info(f"✓ 评估完成，处理了 {len(true_labels)} 个样本")

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # 计算指标
    rst_p, rst_r, rst_f1, _ = precision_recall_fscore_support(
        true_labels, rst_preds, average='macro', zero_division=0
    )
    rst_acc = accuracy_score(true_labels, rst_preds)

    llm_p, llm_r, llm_f1, _ = precision_recall_fscore_support(
        true_labels, llm_preds, average='macro', zero_division=0
    )
    llm_acc = accuracy_score(true_labels, llm_preds)

    # OR 投票
    ensemble_or_preds = [1 if (r + l) >= 1 else 0 for r, l in zip(rst_preds, llm_preds)]
    or_p, or_r, or_f1, _ = precision_recall_fscore_support(
        true_labels, ensemble_or_preds, average='macro', zero_division=0
    )
    or_acc = accuracy_score(true_labels, ensemble_or_preds)

    metrics = {
        'rst_rgat': {
            'f1': rst_f1,
            'precision': rst_p,
            'recall': rst_r,
            'accuracy': rst_acc,
            'total_samples': len(true_labels)
        },
        'llm_qwen': {
            'f1': llm_f1,
            'precision': llm_p,
            'recall': llm_r,
            'accuracy': llm_acc
        },
        'ensemble_or': {
            'f1': or_f1,
            'precision': or_p,
            'recall': or_r,
            'accuracy': or_acc
        }
    }

    return {
        'metrics': metrics,
        'detailed_results': detailed_results,
        'true_labels': true_labels,
        'rst_predictions': rst_preds,
        'llm_predictions': llm_preds
    }


if __name__ == "__main__":
    # 配置
    DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"
    CACHE_FILE = "test_evaluation/rst_cache.pkl"
    QWEN_PATH = "/mnt/second/yuanmengying/qwen3-8b/"

    if not VLLM_AVAILABLE:
        logger.error("vLLM 未安装，请运行: pip install vllm")
        sys.exit(1)

    # 加载数据
    logger.info("加载数据...")
    with open(DATA_PATH, 'r') as f:
        all_data = json.load(f)
    test_data = [s for s in all_data if s.get('dataset') == 'test']

    # 加载缓存
    if not Path(CACHE_FILE).exists():
        logger.error(f"缓存文件不存在: {CACHE_FILE}")
        logger.error("请先运行: python evaluate_test_set.py --cache-rst")
        sys.exit(1)

    with open(CACHE_FILE, 'rb') as f:
        rst_cache = pickle.load(f)

    logger.info(f"✓ 加载了 {len(test_data)} 个测试样本")
    logger.info(f"✓ 加载了 {len(rst_cache)} 个缓存结果\n")

    # 运行评估
    eval_result = evaluate_with_vllm(test_data, rst_cache, QWEN_PATH, batch_size=16)

    # 打印结果
    logger.info("="*80)
    logger.info("评估结果")
    logger.info("="*80)
    logger.info(f"\n【RST-RGAT】")
    logger.info(f"  F1: {eval_result['metrics']['rst_rgat']['f1']:.4f}")
    logger.info(f"  Precision: {eval_result['metrics']['rst_rgat']['precision']:.4f}")
    logger.info(f"  Recall: {eval_result['metrics']['rst_rgat']['recall']:.4f}")

    logger.info(f"\n【Qwen (vLLM)】")
    logger.info(f"  F1: {eval_result['metrics']['llm_qwen']['f1']:.4f}")
    logger.info(f"  Precision: {eval_result['metrics']['llm_qwen']['precision']:.4f}")
    logger.info(f"  Recall: {eval_result['metrics']['llm_qwen']['recall']:.4f}")

    logger.info(f"\n【OR投票】")
    logger.info(f"  F1: {eval_result['metrics']['ensemble_or']['f1']:.4f}")
    logger.info(f"  Precision: {eval_result['metrics']['ensemble_or']['precision']:.4f}")
    logger.info(f"  Recall: {eval_result['metrics']['ensemble_or']['recall']:.4f}")

    logger.info("\n✓ 评估完成！")
