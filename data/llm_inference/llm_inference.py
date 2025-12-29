"""
LLM推理脚本：使用llama3和qwen3进行幻觉检测
支持多种部署方式：Ollama服务、HuggingFace直接加载、API调用
"""

import os
import torch
import pandas as pd
import numpy as np
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployType(Enum):
    """模型部署类型"""
    OLLAMA = "ollama"  # 本地Ollama服务
    HUGGINGFACE = "huggingface"  # HuggingFace直接加载
    API = "api"  # API方式（阿里云、OpenAI等）


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str
    deploy_type: ModelDeployType
    model_path: Optional[str] = None  # HuggingFace模型路径或本地路径
    ollama_url: str = "http://localhost:11434"  # Ollama服务地址
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LLMInference:
    """LLM推理类"""

    # 幻觉检测的prompt模板
    HALLUCINATION_PROMPT_TEMPLATE = """You are an expert in detecting hallucinations in generated text.

Given the context and the generated output, determine if the output contains hallucinations (statements that are not supported by or contradicted by the context).

Context: {context}

Generated Output: {output}

Task: Determine if the output contains hallucinations.
- Output "hallucination" if the output contains hallucinations (statements not supported by context)
- Output "no_hallucination" if the output is factually consistent with the context

Answer (only output 'hallucination' or 'no_hallucination'): """

    # 中文版本的prompt
    HALLUCINATION_PROMPT_ZH = """你是一位专门检测生成文本中幻觉的专家。

幻觉是指在生成的文本中出现的、在给定上下文中没有依据或与上下文矛盾的语句。

上下文：{context}

生成的输出：{output}

任务：判断输出是否包含幻觉。
- 如果输出中有不符合上下文的语句，输出"hallucination"
- 如果输出在事实上与上下文一致，输出"no_hallucination"

答案（只输出'hallucination'或'no_hallucination'）："""

    def __init__(self, config: LLMConfig):
        """初始化LLM推理器"""
        self.config = config
        self.model = None
        self.tokenizer = None

        if config.deploy_type == ModelDeployType.HUGGINGFACE:
            self._load_huggingface_model()
        elif config.deploy_type == ModelDeployType.OLLAMA:
            self._init_ollama()
        elif config.deploy_type == ModelDeployType.API:
            self._init_api()

    def _load_huggingface_model(self):
        """加载HuggingFace模型"""
        logger.info(f"正在加载HuggingFace模型: {self.config.model_path}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            # 加载模型，使用quantization减少内存占用
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # load_in_8bit=True  # 8-bit quantization
            )
            self.model.eval()

            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _init_ollama(self):
        """初始化Ollama连接"""
        logger.info(f"连接Ollama服务: {self.config.ollama_url}")
        # 测试连接
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                logger.info(f"Ollama可用模型: {model_names}")
            else:
                logger.warning(f"Ollama服务响应异常: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到Ollama服务 {self.config.ollama_url}")
            logger.info("请确保Ollama已启动: ollama serve")
            raise

    def _init_api(self):
        """初始化API连接"""
        logger.info(f"使用API: {self.config.api_url}")
        if not self.config.api_key:
            logger.warning("未提供API密钥")

    def _get_prompt(self, context: str, output: str, use_zh: bool = False) -> str:
        """生成提示词"""
        if use_zh:
            return self.HALLUCINATION_PROMPT_ZH.format(context=context, output=output)
        else:
            return self.HALLUCINATION_PROMPT_TEMPLATE.format(context=context, output=output)

    def predict_ollama(self, context: str, output: str, use_zh: bool = False) -> Dict:
        """使用Ollama进行推理"""
        prompt = self._get_prompt(context, output, use_zh)

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.config.temperature,
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip().lower()

                # 解析结果
                if 'hallucination' in generated_text:
                    prediction = 1  # 有幻觉
                    confidence = 0.8
                elif 'no_hallucination' in generated_text:
                    prediction = 0  # 无幻觉
                    confidence = 0.8
                else:
                    # 如果模型没有明确输出，尝试其他启发式方法
                    prediction = 1 if 'hallucination' in generated_text else 0
                    confidence = 0.5

                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'raw_output': generated_text
                }
            else:
                logger.error(f"Ollama API错误: {response.status_code}")
                return {'prediction': -1, 'confidence': 0, 'raw_output': ''}

        except Exception as e:
            logger.error(f"Ollama推理错误: {e}")
            return {'prediction': -1, 'confidence': 0, 'raw_output': str(e)}

    def predict_huggingface(self, context: str, output: str, use_zh: bool = False) -> Dict:
        """使用HuggingFace模型进行推理"""
        prompt = self._get_prompt(context, output, use_zh)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            eos_id = self.model.config.eos_token_id
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    pad_token_id=eos_id,
                    do_sample=False
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取prompt之后的部分
            generated_text = generated_text[len(prompt):].strip().lower()

            # 解析结果
            if 'hallucination' in generated_text:
                prediction = 1
                confidence = 0.8
            elif 'no_hallucination' in generated_text:
                prediction = 0
                confidence = 0.8
            else:
                prediction = 1 if 'hallucination' in generated_text else 0
                confidence = 0.5

            return {
                'prediction': prediction,
                'confidence': confidence,
                'raw_output': generated_text
            }

        except Exception as e:
            logger.error(f"HuggingFace推理错误: {e}")
            return {'prediction': -1, 'confidence': 0, 'raw_output': str(e)}

    def predict_api(self, context: str, output: str, use_zh: bool = False) -> Dict:
        """使用API进行推理（示例：阿里云Qwen）"""
        prompt = self._get_prompt(context, output, use_zh)

        try:
            # 示例：阿里云Qwen API
            if 'aliyun' in self.config.api_url.lower() or 'dashscope' in self.config.api_url.lower():
                return self._predict_aliyun_qwen(prompt)
            # 示例：OpenAI API（支持llama）
            elif 'openai' in self.config.api_url.lower():
                return self._predict_openai(prompt)
            else:
                logger.error(f"不支持的API: {self.config.api_url}")
                return {'prediction': -1, 'confidence': 0, 'raw_output': ''}

        except Exception as e:
            logger.error(f"API推理错误: {e}")
            return {'prediction': -1, 'confidence': 0, 'raw_output': str(e)}

    def _predict_aliyun_qwen(self, prompt: str) -> Dict:
        """调用阿里云Qwen API"""
        try:
            from dashscope import Generation

            response = Generation.call(
                model="qwen-max",  # 或 qwen-plus, qwen-turbo等
                messages=[{'role': 'user', 'content': prompt}],
                api_key=self.config.api_key,
                temperature=self.config.temperature,
            )

            if response.status_code == 200:
                generated_text = response.output.text.lower()

                if 'hallucination' in generated_text:
                    prediction = 1
                    confidence = 0.8
                elif 'no_hallucination' in generated_text:
                    prediction = 0
                    confidence = 0.8
                else:
                    prediction = 1 if 'hallucination' in generated_text else 0
                    confidence = 0.5

                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'raw_output': generated_text
                }
            else:
                logger.error(f"阿里云API错误: {response.message}")
                return {'prediction': -1, 'confidence': 0, 'raw_output': ''}

        except ImportError:
            logger.error("请安装dashscope库: pip install dashscope")
            raise

    def _predict_openai(self, prompt: str) -> Dict:
        """调用OpenAI兼容的API"""
        try:
            import openai

            openai.api_key = self.config.api_key
            openai.api_base = self.config.api_url

            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            generated_text = response.choices[0].message.content.lower()

            if 'hallucination' in generated_text:
                prediction = 1
                confidence = 0.8
            elif 'no_hallucination' in generated_text:
                prediction = 0
                confidence = 0.8
            else:
                prediction = 1 if 'hallucination' in generated_text else 0
                confidence = 0.5

            return {
                'prediction': prediction,
                'confidence': confidence,
                'raw_output': generated_text
            }

        except ImportError:
            logger.error("请安装openai库: pip install openai")
            raise

    def predict(self, context: str, output: str, use_zh: bool = False) -> Dict:
        """根据配置的部署方式进行推理"""
        if self.config.deploy_type == ModelDeployType.OLLAMA:
            return self.predict_ollama(context, output, use_zh)
        elif self.config.deploy_type == ModelDeployType.HUGGINGFACE:
            return self.predict_huggingface(context, output, use_zh)
        elif self.config.deploy_type == ModelDeployType.API:
            return self.predict_api(context, output, use_zh)
        else:
            raise ValueError(f"未知的部署类型: {self.config.deploy_type}")


def load_test_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """加载测试数据"""
    logger.info(f"正在加载测试数据: {data_path}")

    full_df = pd.read_excel(data_path, sheet_name='NLI数据集')
    test_df = full_df[full_df['split'] == 'test'].reset_index(drop=True)

    logger.info(f"测试集大小: {len(test_df)}")

    return test_df


def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """计算详细的评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'no_hallucination': {
            'precision': precision[0],
            'recall': recall[0],
            'f1_score': f1[0],
            'support': int(support[0])
        },
        'hallucination': {
            'precision': precision[1],
            'recall': recall[1],
            'f1_score': f1[1],
            'support': int(support[1])
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'specificity': specificity,
        'sensitivity': sensitivity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
    }


def run_inference(
    model_config: LLMConfig,
    data_path: str,
    output_dir: str = './llm_results',
    use_zh_prompt: bool = False,
    sample_size: Optional[int] = None
):
    """运行推理"""

    os.makedirs(output_dir, exist_ok=True)

    # 初始化推理器
    logger.info(f"初始化{model_config.model_name}推理器...")
    inferencer = LLMInference(model_config)

    # 加载数据
    test_df = load_test_data(data_path)

    # 如果指定了样本数，则抽样
    if sample_size and sample_size < len(test_df):
        logger.info(f"从{len(test_df)}个样本中随机抽取{sample_size}个")
        test_df = test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # 进行推理
    logger.info(f"开始推理（共{len(test_df)}个样本）...")
    predictions = []
    confidences = []
    raw_outputs = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        result = inferencer.predict(row['context'], row['output'], use_zh=use_zh_prompt)
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])
        raw_outputs.append(result['raw_output'])

    # 转换为numpy数组
    y_true = test_df['label'].values
    y_pred = np.array(predictions)

    # 过滤出有效预测（排除-1）
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    logger.info(f"有效预测: {valid_mask.sum()}/{len(test_df)}")

    # 计算指标
    if len(y_pred_valid) > 0:
        metrics = calculate_detailed_metrics(y_true_valid, y_pred_valid)
    else:
        logger.error("没有有效的预测结果")
        return

    # 生成分类报告
    class_report = classification_report(
        y_true_valid, y_pred_valid,
        target_names=['No Hallucination', 'Hallucination'],
        output_dict=True
    )

    # 保存结果
    results = {
        'model_name': model_config.model_name,
        'deploy_type': model_config.deploy_type.value,
        'inference_time': datetime.now().isoformat(),
        'test_size': len(test_df),
        'valid_predictions': int(valid_mask.sum()),
        'use_zh_prompt': use_zh_prompt,
        'detailed_metrics': metrics,
        'classification_report': class_report
    }

    with open(f'{output_dir}/llm_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # 保存详细预测结果
    results_df = test_df.copy()
    results_df['llm_prediction'] = y_pred
    results_df['llm_confidence'] = confidences
    results_df['llm_raw_output'] = raw_outputs
    results_df['correct_prediction'] = (y_pred == y_true)

    results_df.to_excel(f'{output_dir}/llm_detailed_predictions.xlsx', index=False)

    # 打印结果
    print("\n" + "="*70)
    print(f"{model_config.model_name.upper()} 幻觉检测推理结果")
    print("="*70)

    print(f"\n📊 总体性能指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"宏平均精确率: {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
    print(f"宏平均F1分数: {metrics['macro_f1']:.4f}")

    print(f"\n🔍 无幻觉类别 (标签0):")
    no_hall = metrics['no_hallucination']
    print(f"精确率: {no_hall['precision']:.4f}, 召回率: {no_hall['recall']:.4f}, F1: {no_hall['f1_score']:.4f}")

    print(f"\n⚠️  有幻觉类别 (标签1):")
    hall = metrics['hallucination']
    print(f"精确率: {hall['precision']:.4f}, 召回率: {hall['recall']:.4f}, F1: {hall['f1_score']:.4f}")

    print(f"\n📈 关键指标:")
    print(f"敏感性 (Sensitivity): {metrics['sensitivity']:.4f}")
    print(f"特异性 (Specificity): {metrics['specificity']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n🔢 混淆矩阵:")
    print(f"真阴性 (TN): {cm['true_negatives']}, 假阳性 (FP): {cm['false_positives']}")
    print(f"假阴性 (FN): {cm['false_negatives']}, 真阳性 (TP): {cm['true_positives']}")

    print(f"\n💾 结果已保存到: {output_dir}")
    print("="*70)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description='使用LLM进行幻觉检测推理')

    # 模型配置
    parser.add_argument('--model_name', required=True,
                       help='模型名称 (e.g., llama2, qwen, llama2-chat)')
    parser.add_argument('--deploy_type', default='huggingface',
                       choices=['ollama', 'huggingface', 'api'],
                       help='模型部署类型')
    parser.add_argument('--model_path', default=None,
                       help='HuggingFace模型路径或本地模型路径')
    parser.add_argument('--ollama_url', default='http://localhost:11434',
                       help='Ollama服务地址')

    # 数据配置
    parser.add_argument('--data_path', default='../summary_nli_hallucination_dataset.xlsx',
                       help='测试数据路径')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='抽样大小（如果为None则使用全部数据）')

    # 推理配置
    parser.add_argument('--output_dir', default='./llm_results',
                       help='结果保存目录')
    parser.add_argument('--use_zh_prompt', action='store_true',
                       help='使用中文提示词')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='采样温度')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='最大生成token数')

    # API配置
    parser.add_argument('--api_key', default=None,
                       help='API密钥')
    parser.add_argument('--api_url', default=None,
                       help='API地址')

    args = parser.parse_args()

    # 创建配置
    config = LLMConfig(
        model_name=args.model_name,
        deploy_type=ModelDeployType(args.deploy_type),
        model_path=args.model_path,
        ollama_url=args.ollama_url,
        api_key=args.api_key,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # 运行推理
    run_inference(
        model_config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_zh_prompt=args.use_zh_prompt,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
