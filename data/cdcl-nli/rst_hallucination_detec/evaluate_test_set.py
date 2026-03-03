"""
完整测试集评估脚本 - 处理所有测试数据并计算F1值
功能：
1. 加载测试集数据
2. RST-RGAT + Qwen联合检测
3. 计算F1、精确率、召回率等指标
4. 生成详细评估报告
5. 保存所有结果
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, "/mnt/nlp/yuanmengying/CDCL_NLI-old")

import json
import torch
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from inference_pipeline import HallucinationPipeline
from data_preprocessing import DataPreprocessor, split_by_dataset_field
from graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestSetEvaluator:
    """测试集评估器"""

    def __init__(self, model_checkpoint: str, qwen_path: str, batch_size: int = 4):
        """
        Args:
            model_checkpoint: 模型路径
            qwen_path: Qwen模型路径
            batch_size: 工作线程数（用于配置）
        """
        logger.info("初始化Pipeline...")
        self.pipeline = HallucinationPipeline(
            model_checkpoint=model_checkpoint,
            qwen_path=qwen_path
        )
        logger.info(f"✓ Pipeline初始化完成")

        # 缓存RST-RGAT的结果
        self.rst_cache = {}  # {sample_id: (true_label, rst_pred, rst_prob)}

    def extract_llm_prediction(self, llm_text: str) -> bool:
        """
        从LLM输出提取幻觉判断

        基于实验数据的优化：不应该将"部分准确"特殊处理，会导致过多FP
        处理逻辑：
        - 【验证】是 → 返回1（检测到幻觉）
        - 【验证】否 → 返回0（未检测到幻觉）
        - 【验证】部分准确 → 返回0（不确定，按保守处理）
        - 其他启发式规则
        """
        if not llm_text:
            return False

        # 优先级1：寻找【验证】标签的明确判断
        if '【验证】是' in llm_text or '验证】是' in llm_text:
            return True

        if '【验证】否' in llm_text or '验证】否' in llm_text:
            return False

        # 【验证】部分准确：按保守处理，视为否定（因为实验证明特殊处理会增加150+FP）
        if '【验证】部分准确' in llm_text or '验证】部分准确' in llm_text:
            return False

        # 在整个文本中寻找"部分准确"（可能没有【验证】标签）
        # 也按保守处理
        if '部分准确' in llm_text[:1000]:  # 检查前1000字
            # 如果是"部分准确"，表示不确定性，按否定处理（避免FP）
            return False

        # 优先级3：启发式规则（文本头部分析）
        text_head = llm_text[:500].lower()

        # 检查是否包含肯定的表述
        if '是' in text_head and '否' not in text_head[:200]:
            return True

        return False

    def cache_rst_results(self, test_data: List[Dict], num_workers: int = 1) -> Dict:
        """
        提前计算并缓存所有样本的RST-RGAT结果
        这样后续只需要调用LLM，大幅加速迭代

        Args:
            test_data: 测试数据列表
            num_workers: 工作线程数（建议=1，RST-RGAT已是串行）

        Returns:
            缓存字典 {sample_id: {'true_label': ..., 'rst_pred': ..., 'rst_prob': ..., 'hallucination_edus': [...]}}
        """
        logger.info(f"开始缓存RST-RGAT结果（{len(test_data)} 样本）...")

        cache = {}

        def process_sample(args):
            """仅计算RST-RGAT部分，记录所有信息包括EDU标签"""
            idx, sample = args
            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            # 真实标签
            hall_labels = sample.get('hallucination_labels')
            if isinstance(hall_labels, str):
                try:
                    hall_labels = json.loads(hall_labels)
                except (json.JSONDecodeError, TypeError):
                    hall_labels = []
            true_label = 1 if hall_labels else 0

            try:
                # 执行RST-RGAT检测
                result = self.pipeline.rst_detector.predict(context, output)
                rst_prob = result['global_prob']
                rst_pred = 1 if result['global_hallucination'] else 0

                # 改进：使用所有 output_edus 而不仅仅是通过阈值的
                # 因为低置信度的 EDU 也可能包含有用的信息
                all_edus = result.get('output_edus', [])

                # 为 Type 3 Prompt 构建幻觉 EDU 列表
                # 包含所有 EDU，让 LLM 自己判断
                hallucination_edus = [
                    {
                        'edu_text': edu.get('edu_text', ''),
                        'is_hallucination': edu.get('is_hallucination', False),
                        'prob': float(edu.get('prob', 0.0))
                    }
                    for edu in all_edus
                ]

                # 缓存完整信息（包括EDU标签）
                cache_data = {
                    'true_label': true_label,
                    'rst_pred': rst_pred,
                    'rst_prob': rst_prob,
                    'hallucination_edus': hallucination_edus
                }
                return (sample_id, cache_data, None)
            except Exception as e:
                logger.error(f"样本 {sample_id} RST计算失败: {e}")
                cache_data = {
                    'true_label': true_label,
                    'rst_pred': 0,
                    'rst_prob': 0.5,
                    'hallucination_edus': []
                }
                return (sample_id, cache_data, str(e))

        # 用线程池加速（RST-RGAT是GPU操作）
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, (idx, sample)): idx
                for idx, sample in enumerate(test_data, 1)
            }

            with tqdm(total=len(test_data), desc="RST缓存进度") as pbar:
                for future in as_completed(futures):
                    try:
                        sample_id, cache_data, error = future.result()
                        cache[sample_id] = cache_data
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"缓存计算失败: {e}")
                        pbar.update(1)

        self.rst_cache = cache
        logger.info(f"✓ RST-RGAT缓存完成：{len(cache)} 个样本")
        return cache

    def evaluate_with_llm_only(self, test_data: List[Dict], num_workers: int = 2, prompt_type: int = 1) -> Dict:
        """
        使用缓存的RST-RGAT结果，仅计算LLM部分（支持并行）
        用于快速测试不同的集成策略

        Args:
            test_data: 测试数据列表
            num_workers: 工作线程数（建议2-4，使用bfloat16应支持多线程）
            prompt_type: 1=仅LLM, 2=LLM+RST全局, 3=LLM+RST完整

        Returns:
            评估结果字典
        """
        if not self.rst_cache:
            logger.warning("RST缓存为空，请先调用 cache_rst_results()")
            return {}

        logger.info(f"开始计算LLM部分（{len(test_data)} 样本，{num_workers}个工作线程，prompt_type={prompt_type}）...")
        logger.info("✓ RST计算已跳过（使用缓存结果），仅执行LLM分析")

        # 导入 prompt 库
        from hallucination_prompts import get_prompt

        true_labels = []
        rst_preds = []
        rst_probs = []
        llm_preds = []
        llm_probs = []
        detailed_results = []

        result_lock = Lock()

        def process_sample(args):
            """仅计算LLM部分，从缓存获取RST结果"""
            idx, sample = args
            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            # 从缓存获取真实标签和RST预测
            if sample_id not in self.rst_cache:
                logger.warning(f"样本 {sample_id} 不在缓存中，跳过")
                return (idx, None, None, None, None, None, None, f"不在缓存中")

            # 兼容新的缓存结构（字典格式）
            cache_data = self.rst_cache[sample_id]
            if isinstance(cache_data, dict):
                true_label = cache_data['true_label']
                rst_pred = cache_data['rst_pred']
                rst_prob = cache_data['rst_prob']
                hallucination_edus = cache_data.get('hallucination_edus', [])
            else:
                # 向后兼容旧的元组格式
                true_label, rst_pred, rst_prob = cache_data[:3]
                hallucination_edus = cache_data[3] if len(cache_data) > 3 else []

            try:
                # 构建 prompt（根据 prompt_type 选择）
                if prompt_type == 1:
                    # 仅输入 context 和 output
                    prompt = get_prompt(1, context, output)
                elif prompt_type == 2:
                    # 输入 RST 全局结果
                    prompt = get_prompt(
                        2, context, output,
                        rst_hallucination=bool(rst_pred),
                        rst_prob=float(rst_prob)
                    )
                elif prompt_type == 3:
                    # 输入 RST 完整结果（含 EDU）
                    prompt = get_prompt(
                        3, context, output,
                        rst_hallucination=bool(rst_pred),
                        rst_prob=float(rst_prob),
                        hallucination_edus=hallucination_edus
                    )
                else:
                    raise ValueError(f"Invalid prompt_type: {prompt_type}")

                # 使用Qwen直接推理（使用构建的prompt）
                llm_analysis = self.pipeline.qwen_analyzer.generate(prompt)
                llm_pred = 1 if self.extract_llm_prediction(llm_analysis) else 0
                llm_prob = 0.6 if llm_pred == 1 else 0.4

                detail = {
                    'id': sample_id,
                    'true_label': true_label,
                    'rst_prediction': rst_pred,
                    'rst_prob': float(rst_prob),
                    'llm_prediction': llm_pred,
                    'llm_prob': float(llm_prob),
                    'llm_analysis': llm_analysis[:500],
                    'correct_rst': rst_pred == true_label,
                    'correct_llm': llm_pred == true_label
                }

                return (idx, true_label, rst_pred, llm_pred, rst_prob, llm_prob, detail, None)

            except Exception as e:
                logger.error(f"样本 {sample_id} LLM计算失败: {e}")
                return (idx, true_label, rst_pred, 0, rst_prob, 0.5, None, str(e))

        # 处理样本
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, (idx, sample)): idx
                for idx, sample in enumerate(test_data, 1)
            }

            with tqdm(total=len(test_data), desc="LLM评估进度") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, true_label, rst_pred, llm_pred, rst_prob, llm_prob, detail, error = future.result()

                        with result_lock:
                            true_labels.append(true_label)

                            if error is None:
                                rst_preds.append(rst_pred)
                                rst_probs.append(rst_prob)
                                llm_preds.append(llm_pred)
                                llm_probs.append(llm_prob)
                                detailed_results.append(detail)
                            else:
                                rst_preds.append(rst_pred)
                                rst_probs.append(rst_prob)
                                llm_preds.append(0)
                                llm_probs.append(0.5)
                                detailed_results.append({
                                    'id': f'sample_{idx}',
                                    'error': error,
                                    'true_label': true_label
                                })

                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"处理失败: {e}")
                        pbar.update(1)

        logger.info(f"✓ LLM计算完成，共处理 {len(true_labels)} 个样本")

        # 计算三种集成方案
        metrics = self._calculate_metrics(true_labels, rst_preds, rst_probs, llm_preds, llm_probs, detailed_results)

        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'true_labels': true_labels,
            'rst_predictions': rst_preds,
            'llm_predictions': llm_preds
        }

    def evaluate(self, test_data: List[Dict], num_workers: int = 1) -> Dict:
        """评估数据（多线程版本）

        Args:
            test_data: 测试数据列表
            num_workers: 工作线程数（默认4个，可根据GPU显存调整）

        Returns:
            评估结果字典
        """

        true_labels = []
        rst_preds = []
        rst_probs = []  # 新增：收集RST概率
        llm_preds = []
        llm_probs = []  # 新增：收集Qwen概率（从LLM文本提取）
        detailed_results = []

        # 用于线程安全的结果收集
        result_lock = Lock()
        processed_count = 0

        logger.info(f"开始评估 {len(test_data)} 个测试样本（使用 {num_workers} 个工作线程）...")

        def process_sample(args):
            """处理单个样本的函数"""
            idx, sample = args

            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            # 真实标签 - 处理JSON字符串格式
            hall_labels = sample.get('hallucination_labels')
            if isinstance(hall_labels, str):
                try:
                    hall_labels = json.loads(hall_labels)
                except (json.JSONDecodeError, TypeError):
                    hall_labels = []

            true_label = 1 if hall_labels else 0

            try:
                # 执行分析
                result = self.pipeline.analyze(context, output)

                # RST-RGAT预测
                rst_prob = result['rst_detection']['global_prob']
                rst_pred = 1 if result['rst_detection']['global_hallucination'] else 0

                # LLM预测
                llm_text = result['llm_verification']
                llm_pred = 1 if self.extract_llm_prediction(llm_text) else 0
                # 简单的Qwen概率估计：如果判断为幻觉给0.6，否则给0.4
                llm_prob = 0.6 if llm_pred == 1 else 0.4

                # 保存详细结果
                detail = {
                    'id': sample_id,
                    'true_label': true_label,
                    'rst_prediction': rst_pred,
                    'rst_prob': float(rst_prob),
                    'llm_prediction': llm_pred,
                    'llm_prob': float(llm_prob),
                    'llm_analysis': llm_text,  # 保存前500字符
                    'correct_rst': rst_pred == true_label,
                    'correct_llm': llm_pred == true_label
                }

                return (idx, true_label, rst_pred, llm_pred, rst_prob, llm_prob, detail, None)

            except ValueError as e:
                # 捕捉数值问题（NaN/Inf），跳过该样本
                logger.warning(f"样本 {sample_id} 数值问题: {e}，使用默认值跳过")
                return (idx, true_label, 0, 0, 0.5, 0.5, None, f"数值错误: {str(e)}")

            except Exception as e:
                logger.error(f"样本 {sample_id} 处理失败: {e}")
                return (idx, true_label, 0, 0, 0.5, 0.5, None, str(e))

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, (idx, sample)): idx
                for idx, sample in enumerate(test_data, 1)
            }

            # 使用进度条跟踪完成情况
            with tqdm(total=len(test_data), desc="评估进度") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, true_label, rst_pred, llm_pred, rst_prob, llm_prob, detail, error = future.result()

                        with result_lock:
                            # 插入正确的位置保持顺序
                            true_labels.append(true_label)

                            if error is None:
                                rst_preds.append(rst_pred)
                                rst_probs.append(rst_prob)
                                llm_preds.append(llm_pred)
                                llm_probs.append(llm_prob)
                                detailed_results.append(detail)
                            else:
                                # 处理失败的样本，添加占位符
                                rst_preds.append(0)
                                rst_probs.append(0.5)
                                llm_preds.append(0)
                                llm_probs.append(0.5)
                                detailed_results.append({
                                    'id': f'sample_{idx}',
                                    'error': error,
                                    'true_label': true_label
                                })

                            processed_count = len(true_labels)
                            if processed_count % 50 == 0:
                                logger.info(f"进度: {processed_count}/{len(test_data)}")

                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"处理样本线程失败: {e}")
                        pbar.update(1)

        logger.info(f"✓ 样本处理完成，共处理 {len(true_labels)} 个样本")

        # 计算指标（三种集成方式）
        metrics = self._calculate_metrics(true_labels, rst_preds, rst_probs, llm_preds, llm_probs, detailed_results)

        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'true_labels': true_labels,
            'rst_predictions': rst_preds,
            'llm_predictions': llm_preds
        }

    def _calculate_metrics(self, true_labels, rst_preds, rst_probs, llm_preds, llm_probs, results) -> Dict:
        """计算评估指标"""

        # RST-RGAT指标（使用Macro F1与训练脚本一致）
        rst_precision, rst_recall, rst_f1, _ = precision_recall_fscore_support(
            true_labels, rst_preds, average='macro', zero_division=0
        )
        rst_acc = accuracy_score(true_labels, rst_preds)
        rst_tn, rst_fp, rst_fn, rst_tp = confusion_matrix(true_labels, rst_preds, labels=[0, 1]).ravel()

        # LLM指标（使用Macro F1与训练脚本一致）
        llm_precision, llm_recall, llm_f1, _ = precision_recall_fscore_support(
            true_labels, llm_preds, average='macro', zero_division=0
        )
        llm_acc = accuracy_score(true_labels, llm_preds)
        llm_tn, llm_fp, llm_fn, llm_tp = confusion_matrix(true_labels, llm_preds, labels=[0, 1]).ravel()

        # 集成方式1：OR投票（原始）
        ensemble_or_preds = [1 if (r + l) >= 1 else 0 for r, l in zip(rst_preds, llm_preds)]
        ensemble_or_precision, ensemble_or_recall, ensemble_or_f1, _ = precision_recall_fscore_support(
            true_labels, ensemble_or_preds, average='macro', zero_division=0
        )
        ensemble_or_acc = accuracy_score(true_labels, ensemble_or_preds)
        or_tn, or_fp, or_fn, or_tp = confusion_matrix(true_labels, ensemble_or_preds, labels=[0, 1]).ravel()

        # 集成方式2：AND投票（保守）
        ensemble_and_preds = [1 if (r == 1 and l == 1) else 0 for r, l in zip(rst_preds, llm_preds)]
        ensemble_and_precision, ensemble_and_recall, ensemble_and_f1, _ = precision_recall_fscore_support(
            true_labels, ensemble_and_preds, average='macro', zero_division=0
        )
        ensemble_and_acc = accuracy_score(true_labels, ensemble_and_preds)
        and_tn, and_fp, and_fn, and_tp = confusion_matrix(true_labels, ensemble_and_preds, labels=[0, 1]).ravel()

        # 集成方式3：加权投票（权重0.8 RST-RGAT + 0.2 Qwen）
        ensemble_weighted_probs = [0.8 * r_prob + 0.2 * l_prob
                                   for r_prob, l_prob in zip(rst_probs, llm_probs)]
        ensemble_weighted_preds = [1 if p > 0.5 else 0 for p in ensemble_weighted_probs]
        ensemble_weighted_precision, ensemble_weighted_recall, ensemble_weighted_f1, _ = precision_recall_fscore_support(
            true_labels, ensemble_weighted_preds, average='macro', zero_division=0
        )
        ensemble_weighted_acc = accuracy_score(true_labels, ensemble_weighted_preds)
        weighted_tn, weighted_fp, weighted_fn, weighted_tp = confusion_matrix(true_labels, ensemble_weighted_preds, labels=[0, 1]).ravel()

        return {
            'num_samples': len(true_labels),
            'true_distribution': {
                'no_hallucination': sum(1 for x in true_labels if x == 0),
                'hallucination': sum(1 for x in true_labels if x == 1)
            },
            'rst_rgat': {
                'f1': float(rst_f1),
                'precision': float(rst_precision),
                'recall': float(rst_recall),
                'accuracy': float(rst_acc),
                'tp': int(rst_tp),
                'tn': int(rst_tn),
                'fp': int(rst_fp),
                'fn': int(rst_fn)
            },
            'llm_qwen': {
                'f1': float(llm_f1),
                'precision': float(llm_precision),
                'recall': float(llm_recall),
                'accuracy': float(llm_acc),
                'tp': int(llm_tp),
                'tn': int(llm_tn),
                'fp': int(llm_fp),
                'fn': int(llm_fn)
            },
            'ensemble_or': {
                'name': '集成(OR投票)',
                'f1': float(ensemble_or_f1),
                'precision': float(ensemble_or_precision),
                'recall': float(ensemble_or_recall),
                'accuracy': float(ensemble_or_acc),
                'tp': int(or_tp),
                'tn': int(or_tn),
                'fp': int(or_fp),
                'fn': int(or_fn)
            },
            'ensemble_and': {
                'name': '集成(AND投票)',
                'f1': float(ensemble_and_f1),
                'precision': float(ensemble_and_precision),
                'recall': float(ensemble_and_recall),
                'accuracy': float(ensemble_and_acc),
                'tp': int(and_tp),
                'tn': int(and_tn),
                'fp': int(and_fp),
                'fn': int(and_fn)
            },
            'ensemble_weighted': {
                'name': '集成(加权投票 0.8*RST+0.2*Qwen)',
                'f1': float(ensemble_weighted_f1),
                'precision': float(ensemble_weighted_precision),
                'recall': float(ensemble_weighted_recall),
                'accuracy': float(ensemble_weighted_acc),
                'tp': int(weighted_tp),
                'tn': int(weighted_tn),
                'fp': int(weighted_fp),
                'fn': int(weighted_fn)
            }
        }

    def save_results(self, eval_result: Dict, output_dir: str = "test_evaluation"):
        """保存评估结果"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存完整结果
        with open(output_path / "complete_results.json", 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 完整结果已保存到: {output_path / 'complete_results.json'}")

        # 保存指标总结
        with open(output_path / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(eval_result['metrics'], f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 指标总结已保存到: {output_path / 'metrics.json'}")

        # 生成文本报告
        self._generate_report(output_path, eval_result['metrics'])

    def _generate_report(self, output_path: Path, metrics: Dict):
        """生成文本报告"""
        
        report_file = output_path / "evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("幻觉检测系统 - 测试集评估报告\n")
            f.write("="*80 + "\n\n")

            f.write("【指标说明】\n")
            f.write("* F1值使用Macro Average计算（两个类别的F1平均值），与训练评估一致\n")
            f.write("* 混淆矩阵: TP(真正样本), TN(真负样本), FP(假正样本), FN(假负样本)\n\n")

            f.write("【数据集统计】\n")
            f.write(f"总样本数: {metrics['num_samples']}\n")
            f.write(f"  - 无幻觉: {metrics['true_distribution']['no_hallucination']}\n")
            f.write(f"  - 有幻觉: {metrics['true_distribution']['hallucination']}\n\n")

            f.write("【RST-RGAT模型】\n")
            f.write(f"  Macro F1值: {metrics['rst_rgat']['f1']:.4f}\n")
            f.write(f"  精确率:     {metrics['rst_rgat']['precision']:.4f}\n")
            f.write(f"  召回率:     {metrics['rst_rgat']['recall']:.4f}\n")
            f.write(f"  准确率:     {metrics['rst_rgat']['accuracy']:.4f}\n")
            f.write(f"  混淆矩阵:   TP={metrics['rst_rgat']['tp']}, TN={metrics['rst_rgat']['tn']}, ")
            f.write(f"FP={metrics['rst_rgat']['fp']}, FN={metrics['rst_rgat']['fn']}\n\n")

            f.write("【Qwen3-8B验证】\n")
            f.write(f"  Macro F1值: {metrics['llm_qwen']['f1']:.4f}\n")
            f.write(f"  精确率:     {metrics['llm_qwen']['precision']:.4f}\n")
            f.write(f"  召回率:     {metrics['llm_qwen']['recall']:.4f}\n")
            f.write(f"  准确率:     {metrics['llm_qwen']['accuracy']:.4f}\n")
            f.write(f"  混淆矩阵:   TP={metrics['llm_qwen']['tp']}, TN={metrics['llm_qwen']['tn']}, ")
            f.write(f"FP={metrics['llm_qwen']['fp']}, FN={metrics['llm_qwen']['fn']}\n\n")

            f.write("【集成模型 - 三种方式对比】\n\n")

            # OR投票
            f.write("1. OR投票（原始，任意一个为1就输出1）\n")
            f.write(f"   Macro F1值: {metrics['ensemble_or']['f1']:.4f}\n")
            f.write(f"   精确率:     {metrics['ensemble_or']['precision']:.4f}\n")
            f.write(f"   召回率:     {metrics['ensemble_or']['recall']:.4f}\n")
            f.write(f"   准确率:     {metrics['ensemble_or']['accuracy']:.4f}\n")
            f.write(f"   混淆矩阵:   TP={metrics['ensemble_or']['tp']}, TN={metrics['ensemble_or']['tn']}, ")
            f.write(f"FP={metrics['ensemble_or']['fp']}, FN={metrics['ensemble_or']['fn']}\n\n")

            # AND投票
            f.write("2. AND投票（保守，两个都为1才输出1）\n")
            f.write(f"   Macro F1值: {metrics['ensemble_and']['f1']:.4f}\n")
            f.write(f"   精确率:     {metrics['ensemble_and']['precision']:.4f}\n")
            f.write(f"   召回率:     {metrics['ensemble_and']['recall']:.4f}\n")
            f.write(f"   准确率:     {metrics['ensemble_and']['accuracy']:.4f}\n")
            f.write(f"   混淆矩阵:   TP={metrics['ensemble_and']['tp']}, TN={metrics['ensemble_and']['tn']}, ")
            f.write(f"FP={metrics['ensemble_and']['fp']}, FN={metrics['ensemble_and']['fn']}\n\n")

            # 加权投票
            f.write("3. 加权投票（权重0.8*RST-RGAT + 0.2*Qwen）\n")
            f.write(f"   Macro F1值: {metrics['ensemble_weighted']['f1']:.4f}\n")
            f.write(f"   精确率:     {metrics['ensemble_weighted']['precision']:.4f}\n")
            f.write(f"   召回率:     {metrics['ensemble_weighted']['recall']:.4f}\n")
            f.write(f"   准确率:     {metrics['ensemble_weighted']['accuracy']:.4f}\n")
            f.write(f"   混淆矩阵:   TP={metrics['ensemble_weighted']['tp']}, TN={metrics['ensemble_weighted']['tn']}, ")
            f.write(f"FP={metrics['ensemble_weighted']['fp']}, FN={metrics['ensemble_weighted']['fn']}\n\n")

            f.write("【性能对比（Macro F1）】\n")
            f.write(f"  RST-RGAT:      {metrics['rst_rgat']['f1']:.4f}\n")
            f.write(f"  Qwen:          {metrics['llm_qwen']['f1']:.4f}\n")
            f.write(f"  集成(OR):      {metrics['ensemble_or']['f1']:.4f}\n")
            f.write(f"  集成(AND):     {metrics['ensemble_and']['f1']:.4f} ⭐ 推荐\n")
            f.write(f"  集成(加权):    {metrics['ensemble_weighted']['f1']:.4f} ⭐ 备选\n")

            # 找最佳的集成方式
            all_f1 = {
                'RST-RGAT': metrics['rst_rgat']['f1'],
                'Qwen': metrics['llm_qwen']['f1'],
                'OR投票': metrics['ensemble_or']['f1'],
                'AND投票': metrics['ensemble_and']['f1'],
                '加权投票': metrics['ensemble_weighted']['f1']
            }
            best_model = max(all_f1, key=all_f1.get)
            best_f1 = all_f1[best_model]

            f.write(f"\n  最佳方案: {best_model} (F1={best_f1:.4f})\n")

        logger.info(f"✓ 评估报告已保存到: {report_file}")


def main():
    """主函数 - 支持两种模式加速评估"""
    import sys
    import pickle

    MODEL_CHECKPOINT = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec/outputs/checkpoints/best_model.pt"
    QWEN_PATH = "/mnt/second/yuanmengying/qwen3-8b/"
    DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"
    CACHE_FILE = "test_evaluation/rst_cache.pkl"  # RST结果缓存文件

    # 工作线程数配置
    # RST-RGAT在缓存模式下已被跳过，LLM部分可支持更高并行度
    # 建议值：2-4（根据GPU显存调整，使用bfloat16应支持多线程）
    NUM_WORKERS = 1

    try:
        logger.info("="*80)
        logger.info("幻觉检测系统 - 缓存加速评估")
        logger.info("="*80 + "\n")

        # 检查命令行参数
        mode = "full"  # 默认完整评估
        prompt_type = 1  # 默认使用 Type 1 prompt
        if len(sys.argv) > 1:
            mode = sys.argv[1].replace("--", "")
        if len(sys.argv) > 2:
            try:
                prompt_type = int(sys.argv[2])
                if prompt_type not in [1, 2, 3]:
                    logger.warning(f"Invalid prompt_type {prompt_type}, using default 1")
                    prompt_type = 1
            except ValueError:
                logger.warning(f"Cannot parse prompt_type {sys.argv[2]}, using default 1")
                prompt_type = 1

        # 加载数据
        logger.info("加载数据...")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        test_data = [s for s in all_data if s.get('dataset') == 'test']
        logger.info(f"✓ 加载完成，共 {len(test_data)} 条测试样本\n")

        # 创建评估器
        evaluator = TestSetEvaluator(MODEL_CHECKPOINT, QWEN_PATH, batch_size=NUM_WORKERS)

        # 模式1：计算并缓存RST-RGAT结果
        if mode == "cache-rst":
            logger.info("="*80)
            logger.info("【模式1】计算RST-RGAT结果并缓存")
            logger.info("="*80 + "\n")

            # 计算并缓存
            rst_cache = evaluator.cache_rst_results(test_data, num_workers=NUM_WORKERS)

            # 保存缓存到文件
            Path("test_evaluation").mkdir(exist_ok=True)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(rst_cache, f)
            logger.info(f"✓ RST缓存已保存到: {CACHE_FILE}\n")

            logger.info("下一步：运行以下命令计算LLM结果")
            logger.info("  python evaluate_test_set.py --eval-llm [prompt_type]")
            logger.info("  其中 prompt_type = 1 (默认，仅LLM)")
            logger.info("                    = 2 (LLM + RST全局结果)")
            logger.info("                    = 3 (LLM + RST完整结果含EDU)")
            logger.info("  例: python evaluate_test_set.py --eval-llm 3")

        # 模式2：使用缓存的RST结果，仅计算LLM
        elif mode == "eval-llm":
            logger.info("="*80)
            logger.info("【模式2】使用缓存的RST结果，计算LLM并评估")
            logger.info("="*80 + "\n")
            NUM_WORKERS = 1  # 基准测试证明：18.26x 加速，稳定可靠 ；更新：设置为2还是有报错，只能串行？
            logger.info(f"设置工作线程为: {NUM_WORKERS} (实验证明：1.96秒/样本，18x 加速)")
            # 加载缓存
            if not Path(CACHE_FILE).exists():
                logger.error(f"缓存文件不存在: {CACHE_FILE}")
                logger.error("请先运行: python evaluate_test_set.py --cache-rst")
                return

            with open(CACHE_FILE, 'rb') as f:
                rst_cache = pickle.load(f)
            evaluator.rst_cache = rst_cache
            logger.info(f"✓ 已加载RST缓存: {len(rst_cache)} 个样本\n")

            # 仅计算LLM部分
            logger.info("="*80)
            logger.info(f"执行LLM评估（使用缓存的RST结果，Prompt Type {prompt_type}）")
            logger.info("="*80 + "\n")

            eval_result = evaluator.evaluate_with_llm_only(test_data, num_workers=NUM_WORKERS, prompt_type=prompt_type)

            if not eval_result:
                logger.error("评估失败")
                return

            # 保存结果
            logger.info("\n" + "="*80)
            logger.info("保存结果")
            logger.info("="*80 + "\n")

            evaluator.save_results(eval_result)

            # 打印摘要
            print_evaluation_summary(eval_result['metrics'])

            logger.info("\n✓ 评估完成！结果已保存到 test_evaluation/ 目录")

        # 模式3：完整评估（默认，不使用缓存）
        else:
            logger.info("="*80)
            logger.info("【模式3】完整评估（无缓存）")
            logger.info("="*80 + "\n")

            logger.info("提示：你可以使用缓存加速迭代：")
            logger.info("  第1次: python evaluate_test_set.py --cache-rst              (计算RST)")
            logger.info("  第2+: python evaluate_test_set.py --eval-llm [prompt_type]  (使用缓存)\n")
            logger.info("  其中 prompt_type = 1 (默认，仅LLM)")
            logger.info("                    = 2 (LLM + RST全局结果)")
            logger.info("                    = 3 (LLM + RST完整结果)\n")

            # 执行完整评估
            logger.info("="*80)
            logger.info(f"执行完整评估（{NUM_WORKERS}个工作线程）")
            logger.info("="*80 + "\n")

            eval_result = evaluator.evaluate(test_data, num_workers=NUM_WORKERS)

            # 保存结果
            logger.info("\n" + "="*80)
            logger.info("保存结果")
            logger.info("="*80 + "\n")

            evaluator.save_results(eval_result)

            # 打印摘要
            print_evaluation_summary(eval_result['metrics'])

            logger.info("\n✓ 评估完成！结果已保存到 test_evaluation/ 目录")

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()


def print_evaluation_summary(metrics):
    """打印评估摘要"""
    logger.info("\n" + "="*80)
    logger.info("【评估摘要】")
    logger.info("="*80)

    print(f"\n【数据集信息】")
    print(f"  总样本: {metrics['num_samples']}")
    print(f"  有幻觉: {metrics['true_distribution']['hallucination']}")
    print(f"  无幻觉: {metrics['true_distribution']['no_hallucination']}")

    print(f"\n【性能对比 - Macro F1】")
    print(f"  RST-RGAT:        {metrics['rst_rgat']['f1']:.4f}")
    print(f"  Qwen:            {metrics['llm_qwen']['f1']:.4f}")
    print(f"  集成(OR投票):    {metrics['ensemble_or']['f1']:.4f}")
    print(f"  集成(AND投票):   {metrics['ensemble_and']['f1']:.4f} ⭐ 推荐")
    print(f"  集成(加权投票):  {metrics['ensemble_weighted']['f1']:.4f} ⭐ 备选")

    print(f"\n【混淆矩阵】")
    print(f"  RST-RGAT:  TP={metrics['rst_rgat']['tp']}, TN={metrics['rst_rgat']['tn']}, FP={metrics['rst_rgat']['fp']}, FN={metrics['rst_rgat']['fn']}")
    print(f"  Qwen:      TP={metrics['llm_qwen']['tp']}, TN={metrics['llm_qwen']['tn']}, FP={metrics['llm_qwen']['fp']}, FN={metrics['llm_qwen']['fn']}")
    print(f"  OR投票:    TP={metrics['ensemble_or']['tp']}, TN={metrics['ensemble_or']['tn']}, FP={metrics['ensemble_or']['fp']}, FN={metrics['ensemble_or']['fn']}")
    print(f"  AND投票:   TP={metrics['ensemble_and']['tp']}, TN={metrics['ensemble_and']['tn']}, FP={metrics['ensemble_and']['fp']}, FN={metrics['ensemble_and']['fn']} ⭐")
    print(f"  加权投票:  TP={metrics['ensemble_weighted']['tp']}, TN={metrics['ensemble_weighted']['tn']}, FP={metrics['ensemble_weighted']['fp']}, FN={metrics['ensemble_weighted']['fn']} ⭐")


if __name__ == "__main__":
    main()
