#!/usr/bin/env python3
"""
多模型评估脚本 - 对比不同 LLM 在幻觉检测任务中的性能
使用 RST-RGAT 缓存结果，仅评估 LLM 部分
支持 OpenAI 兼容协议的所有模型
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, "/mnt/nlp/yuanmengying/CDCL_NLI-old")

import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from hallucination_prompts import get_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_llm_prediction(llm_text: str) -> bool:
    """保守的LLM判断提取函数"""
    if not llm_text:
        return False
    if '【验证】是' in llm_text or '验证】是' in llm_text:
        return True
    if '【验证】否' in llm_text or '验证】否' in llm_text:
        return False
    if '【验证】部分准确' in llm_text or '验证】部分准确' in llm_text:
        return False
    if '部分准确' in llm_text[:1000]:
        return False
    text_head = llm_text[:500].lower()
    if '是' in text_head and '否' not in text_head[:200]:
        return True
    return False


class MultiModelEvaluator:
    """多模型评估器"""

    def __init__(self, api_key: str, api_endpoint: str):
        """
        初始化评估器

        Args:
            api_key: OpenAI 兼容 API key
            api_endpoint: API endpoint (如：https://coding.dashscope.aliyuncs.com/v1)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_endpoint
        )
        self.api_endpoint = api_endpoint
        logger.info(f"✓ Dashscope 客户端初始化完成")
        logger.info(f"  Endpoint: {api_endpoint}")

    def _build_prompt_by_type(self, context: str, output: str, detection_result: Dict, prompt_type: int = 1) -> str:
        """根据 prompt_type 构建评估 prompt"""
        global_hallucination = detection_result['global_hallucination']
        global_prob = detection_result['global_prob']
        hallucination_edus = detection_result.get('hallucination_edus', [])

        if prompt_type == 1:
            # 仅使用 context 和 output
            return get_prompt(1, context, output)
        elif prompt_type == 2:
            # 使用 RST 全局结果
            return get_prompt(
                2, context, output,
                rst_hallucination=global_hallucination,
                rst_prob=global_prob
            )
        elif prompt_type == 3:
            # 使用 RST 完整结果（含 EDU）
            return get_prompt(
                3, context, output,
                rst_hallucination=global_hallucination,
                rst_prob=global_prob,
                hallucination_edus=hallucination_edus
            )
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")

    def evaluate_model(
        self,
        model_name: str,
        test_data: List[Dict],
        rst_cache: Dict,
        prompt_type: int = 1,
        num_workers: int = 1
    ) -> Dict:
        """
        评估单个模型

        Args:
            model_name: 模型名称（如 qwen3.5-plus）
            test_data: 测试数据
            rst_cache: RST 缓存
            prompt_type: 1=仅LLM, 2=LLM+RST全局, 3=LLM+RST完整
            num_workers: 工作线程数

        Returns:
            评估结果字典
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"评估模型: {model_name} (Prompt Type {prompt_type})")
        logger.info(f"{'='*80}")

        true_labels = []
        rst_preds = []
        rst_probs = []
        llm_preds = []
        llm_probs = []
        errors = []

        result_lock = Lock()
        start_time = time.time()
        processed_count = [0]  # 用列表以便在内部函数中修改

        def process_sample(args):
            """处理单个样本"""
            idx, sample = args
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

            # 从缓存获取 RST 结果（兼容新的dict格式和旧的tuple格式）
            if sample_id not in rst_cache:
                return (idx, true_label, 0, 0.5, 0, 0.5, None, f"样本 {sample_id} 不在缓存中")

            cache_data = rst_cache[sample_id]
            if isinstance(cache_data, dict):
                true_label_cached = cache_data['true_label']
                rst_pred = cache_data['rst_pred']
                rst_prob = cache_data['rst_prob']
                hallucination_edus = cache_data.get('hallucination_edus', [])
            else:
                # 向后兼容旧的元组格式
                true_label_cached, rst_pred, rst_prob = cache_data[:3]
                hallucination_edus = cache_data[3] if len(cache_data) > 3 else []

            try:
                # 构建 prompt（使用新的 get_prompt 系统）
                mock_detection_result = {
                    'global_hallucination': bool(rst_pred),
                    'global_prob': float(rst_prob),
                    'hallucination_edus': hallucination_edus
                }
                prompt = self._build_prompt_by_type(context, output, mock_detection_result, prompt_type)

                # 调用 LLM（带重试）
                max_retries = 3
                retry_count = 0
                last_error = None
                response = None

                while retry_count < max_retries:
                    try:
                        # 尝试使用带参数的方式
                        response = self.client.responses.create(
                            model=model_name,
                            input=prompt,
                            temperature=0.7,
                            top_p=0.95,
                            max_tokens=512
                        )
                        break  # 成功，跳出重试循环
                    except TypeError as e:
                        # 如果不支持这些参数，尝试仅用基础参数
                        if "unexpected keyword argument" in str(e):
                            logger.debug(f"样本 {sample_id} 尝试使用最小参数调用")
                            try:
                                response = self.client.responses.create(
                                    model=model_name,
                                    input=prompt
                                )
                                break
                            except Exception as e2:
                                last_error = e2
                                retry_count += 1
                                if retry_count < max_retries:
                                    wait_time = min(2 ** retry_count, 30)
                                    logger.warning(f"样本 {sample_id} 第 {retry_count} 次重试失败，等待 {wait_time} 秒后重试: {str(e2)[:100]}")
                                    import time
                                    time.sleep(wait_time)
                        else:
                            raise e
                    except Exception as e:
                        last_error = e
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = min(2 ** retry_count, 30)
                            logger.warning(f"样本 {sample_id} 第 {retry_count} 次重试失败，等待 {wait_time} 秒后重试: {str(e)[:100]}")
                            import time
                            time.sleep(wait_time)

                # 尝试提取输出，如果全部失败使用默认值
                llm_text = ""
                if response is not None:
                    try:
                        llm_text = response.output_text if hasattr(response, 'output_text') else str(response)
                    except Exception as e:
                        logger.warning(f"样本 {sample_id} 解析响应失败: {str(e)[:100]}")
                        llm_text = ""
                else:
                    # API 调用完全失败，使用默认值（空字符串表示无有效响应）
                    logger.warning(f"样本 {sample_id} API 调用全部失败，使用默认值")

                llm_pred = 1 if extract_llm_prediction(llm_text) else 0
                llm_prob = 0.6 if llm_pred == 1 else 0.4

                return (
                    idx,
                    true_label_cached,
                    rst_pred,
                    rst_prob,
                    llm_pred,
                    llm_prob,
                    {
                        'id': sample_id,
                        'true_label': true_label_cached,
                        'rst_pred': rst_pred,
                        'llm_pred': llm_pred,
                        'llm_analysis': llm_text[:200]
                    },
                    None
                )

            except Exception as e:
                # 确保任何失败也返回有效值，而不是完全失败
                error_msg = f"样本 {sample_id} 处理失败: {str(e)[:100]}"
                logger.warning(error_msg)
                # 返回默认值：保持 RST 结果，LLM 预测为 0（保守）
                return (idx, true_label_cached, rst_pred, rst_prob, 0, 0.5, None, error_msg)

        # 使用线程池处理
        logger.info(f"开始评估（{num_workers} 个工作线程）...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, (idx, sample)): idx
                for idx, sample in enumerate(test_data, 1)
            }

            with tqdm(total=len(test_data), desc=f"进度") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, true_label, rst_pred, rst_prob, llm_pred, llm_prob, detail, error = future.result()

                        with result_lock:
                            true_labels.append(true_label)
                            rst_preds.append(rst_pred)
                            rst_probs.append(rst_prob)
                            llm_preds.append(llm_pred)
                            llm_probs.append(llm_prob)

                            if detail:
                                pass  # 可以保存详细结果
                            if error:
                                errors.append(error)

                            processed_count[0] += 1

                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"处理失败: {e}")
                        pbar.update(1)

        elapsed_time = time.time() - start_time

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

        # 打印结果
        logger.info(f"✓ 评估完成")
        logger.info(f"  处理样本: {len(true_labels)}")
        logger.info(f"  错误数: {len(errors)}")
        logger.info(f"  耗时: {elapsed_time:.1f} 秒 ({elapsed_time/len(true_labels):.2f}s/样本)")
        logger.info(f"\n【{model_name} 性能】")
        logger.info(f"  LLM F1:        {llm_f1:.4f}")
        logger.info(f"  LLM Precision: {llm_p:.4f}")
        logger.info(f"  LLM Recall:    {llm_r:.4f}")
        logger.info(f"  LLM Accuracy:  {llm_acc:.4f}")
        logger.info(f"  OR投票 F1:      {or_f1:.4f}")

        return {
            'model': model_name,
            'elapsed_time': elapsed_time,
            'samples_processed': len(true_labels),
            'errors': len(errors),
            'rst_rgat': {
                'f1': rst_f1,
                'precision': rst_p,
                'recall': rst_r,
                'accuracy': rst_acc
            },
            'llm': {
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


def main():
    import os

    # 配置
    API_KEY = "sk-c0d62d2a1cb349b985cc42bd4c56b17d"
    if not API_KEY:
        logger.error("DASHSCOPE_API_KEY 环境变量未设置")
        logger.error("请运行: export DASHSCOPE_API_KEY='your-api-key'")
        sys.exit(1)

    API_ENDPOINT = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"

    # 模型列表
    MODELS = [
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "glm-4.7",
        "kimi-k2.5"
    ]

    DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"
    CACHE_FILE = "test_evaluation/rst_cache.pkl"

    # 支持命令行参数选择 prompt_type
    prompt_type = 1  # 默认使用 Type 1
    if len(sys.argv) > 1:
        try:
            prompt_type = int(sys.argv[1])
            if prompt_type not in [1, 2, 3]:
                logger.warning(f"Invalid prompt_type {prompt_type}, using default 1")
                prompt_type = 1
        except ValueError:
            logger.warning(f"Cannot parse prompt_type {sys.argv[1]}, using default 1")
            prompt_type = 1

    logger.info("\n" + "="*80)
    logger.info("⚠️  性能提示")
    logger.info("="*80)
    logger.info("本脚本使用 OpenAI 兼容 API，响应较慢（平均 ~90-120s/样本）")
    logger.info("如果仅需单一模型评估，建议使用本地 Qwen 模型：")
    logger.info("  python evaluate_test_set.py --eval-llm [prompt_type]")
    logger.info("  这样会快 20-30 倍（~3s/样本）")
    logger.info("\n本脚本适合：对比多个 API 模型的性能")
    logger.info("="*80 + "\n")

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
    logger.info(f"✓ 加载了 {len(rst_cache)} 个缓存结果")

    # 初始化评估器
    evaluator = MultiModelEvaluator(API_KEY, API_ENDPOINT)

    # 评估所有模型
    logger.info("\n" + "="*80)
    logger.info(f"开始多模型评估 (Prompt Type {prompt_type})")
    logger.info("="*80)

    results = {}
    for model in MODELS:
        try:
            result = evaluator.evaluate_model(model, test_data, rst_cache, prompt_type=prompt_type, num_workers=1)
            results[model] = result
        except Exception as e:
            logger.error(f"模型 {model} 评估失败: {e}")
            results[model] = {'error': str(e)}

    # 输出对比总结
    logger.info("\n" + "="*80)
    logger.info(f"多模型对比总结 (Prompt Type {prompt_type})")
    logger.info("="*80 + "\n")

    # 构建对比表格
    print(f"{'模型':<25} | {'LLM F1':<8} | {'精确率':<8} | {'召回率':<8} | {'准确率':<8} | {'耗时(s)':<8}")
    print("-" * 90)

    best_model = None
    best_f1 = -1

    for model in MODELS:
        if 'error' in results[model]:
            print(f"{model:<25} | {'ERROR':<8} | {'':<8} | {'':<8} | {'':<8} | {'':<8}")
        else:
            r = results[model]
            llm_f1 = r['llm']['f1']
            llm_p = r['llm']['precision']
            llm_r = r['llm']['recall']
            llm_acc = r['llm']['accuracy']
            elapsed = r['elapsed_time']

            print(f"{model:<25} | {llm_f1:<8.4f} | {llm_p:<8.4f} | {llm_r:<8.4f} | {llm_acc:<8.4f} | {elapsed:<8.1f}")

            if llm_f1 > best_f1:
                best_f1 = llm_f1
                best_model = model

    print()
    if best_model:
        logger.info(f"✓ 最优模型: {best_model} (F1={best_f1:.4f})")

    # 保存完整结果
    output_file = f"test_evaluation/multi_model_results_prompt_type_{prompt_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 完整结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
