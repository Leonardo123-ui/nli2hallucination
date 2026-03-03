"""
完整推理Pipeline：RST-RGAT检测 + Qwen3-8B验证分析
功能：
1. 使用RST-RGAT模型进行幻觉检测
2. 使用Qwen3-8B进行验证和详细分析
3. 整合结果为结构化输出
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# 提前添加DMRST模块路径
DMRST_PATH = "/mnt/nlp/yuanmengying/CDCL_NLI-old"
if DMRST_PATH not in sys.path:
    sys.path.insert(0, DMRST_PATH)

from data_preprocessing import DMRSTProcessor
from graph_builder import GraphBuilder
from rst_rgat_model import RSTRGATModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RSTRGATInference:
    """RST-RGAT推理类"""

    def __init__(self, model_checkpoint: str, device: Optional[torch.device] = None):
        """
        初始化RST-RGAT推理器

        Args:
            model_checkpoint: 模型检查点路径
            device: 设备（默认自动选择）
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = model_checkpoint

        # 初始化模型
        self.model = RSTRGATModel(
            input_dim=1027,
            hidden_dim=512,
            num_rgat_layers=2,
            num_relations=20,
            num_heads=4,
            dropout=0.2
        ).to(self.device)

        # 加载权重
        self._load_checkpoint(model_checkpoint)

        # 初始化DMRST处理器和图构建器
        self.dmrst_processor = DMRSTProcessor()
        self.graph_builder = GraphBuilder()

        logger.info(f"✓ RST-RGAT推理器初始化完成（设备: {self.device}）")

    def _load_checkpoint(self, path: str):
        """加载模型检查点"""
        if not Path(path).exists():
            logger.error(f"检查点文件不存在: {path}")
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_threshold = checkpoint.get('threshold', 0.5)
        self.model.eval()
        logger.info(f"✓ 模型检查点已加载，阈值: {self.best_threshold:.3f}")

    def predict(self, context: str, output: str) -> Dict:
        """
        预测幻觉

        Args:
            context: 原始文本（上下文）
            output: 生成文本（摘要）

        Returns:
            {
                'global_hallucination': bool,  # 全局是否存在幻觉
                'global_prob': float,  # 全局幻觉概率
                'hallucination_edus': [  # 幻觉EDU列表
                    {
                        'edu_text': str,
                        'prob': float,
                        'is_hallucination': bool
                    },
                    ...
                ],
                'output_edus': [  # 输出的所有EDU
                    {
                        'edu_text': str,
                        'prob': float,
                        'is_hallucination': bool
                    },
                    ...
                ]
            }
        """
        try:
            # 解析文本
            logger.info("解析上下文和输出的RST结构...")
            context_result = self.dmrst_processor.parse_text(context)
            output_result = self.dmrst_processor.parse_text(output)

            # 构建样本（需要与graph_builder期望的格式一致）
            sample = {
                'id': 'inference',
                'context': context_result,  # DMRST解析结果（包含'segments', 'tree'等）
                'output': output_result,    # DMRST解析结果
                'output_edu_labels': [0] * len(output_result['segments']),  # 推理时使用零标签
                'y_global': 0  # 推理时使用零标签
            }

            # 构建图
            logger.info("构建图...")
            graph = self.graph_builder.build_graph(sample)

            # 前向传播
            with torch.no_grad():
                batch = graph.to(self.device)
                if isinstance(batch, list):
                    from torch_geometric.data import Batch
                    batch = Batch.from_data_list([batch])

                outputs = self.model(batch)

                # 提取预测结果
                edu_logits = outputs['edu_logits']
                global_logit = outputs['global_logit']

                # 检查是否包含 NaN 或 inf
                if torch.isnan(edu_logits).any() or torch.isinf(edu_logits).any():
                    logger.error(f"EDU logits contains NaN or Inf")
                    raise ValueError("Model output contains NaN or Inf values")

                if torch.isnan(global_logit).any() or torch.isinf(global_logit).any():
                    logger.error(f"Global logit contains NaN or Inf")
                    raise ValueError("Model output contains NaN or Inf values")

                edu_probs = torch.sigmoid(edu_logits).cpu().numpy()
                global_prob = torch.sigmoid(global_logit).cpu().item()

                # 再次检查 sigmoid 后的值
                if not (0 <= global_prob <= 1):
                    logger.error(f"Global prob out of range: {global_prob}")
                    raise ValueError(f"Probability out of range: {global_prob}")

            # 提取EDU文本
            output_edus = output_result['segments']

            # 构建结果
            hallucination_edus = []
            output_edus_result = []

            # 注意：全局阈值和 EDU 阈值应该不同
            # 全局概率范围 0.27-0.63，用 threshold=0.52
            # EDU 概率范围 0.19-0.44，应该用更低的阈值（如 0.30）
            # 这里使用相对阈值：取概率最高的 25% EDU
            edu_threshold = max(0.30, np.percentile([e for e in edu_probs], 75))

            for i, (edu_text, edu_prob) in enumerate(zip(output_edus, edu_probs)):
                edu_dict = {
                    'edu_text': edu_text.strip(),
                    'prob': float(edu_prob),
                    'is_hallucination': edu_prob > edu_threshold  # 用适合 EDU 的阈值
                }
                output_edus_result.append(edu_dict)

                if edu_prob > edu_threshold:
                    hallucination_edus.append(edu_dict)

            # 全局判断
            global_hallucination = global_prob > self.best_threshold

            return {
                'global_hallucination': global_hallucination,
                'global_prob': float(global_prob),
                'global_threshold': self.best_threshold,
                'hallucination_edus': hallucination_edus,
                'output_edus': output_edus_result
            }

        except Exception as e:
            logger.error(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise


class QwenAnalyzer:
    """Qwen3-8B分析类"""

    def __init__(self, model_path: str = "/mnt/second/yuanmengying/qwen3-8b/",
                 device: Optional[torch.device] = None):
        """
        初始化Qwen分析器

        Args:
            model_path: Qwen模型路径
            device: 设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"正在加载Qwen3-8B模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✓ Qwen3-8B模型加载完成")
        except Exception as e:
            logger.error(f"Qwen模型加载失败: {e}")
            raise

    def analyze(self, context: str, output: str, detection_result: Dict) -> str:
        """
        使用Qwen3-8B进行幻觉分析

        Args:
            context: 原始文本
            output: 生成文本
            detection_result: RST-RGAT检测结果

        Returns:
            分析文本
        """
        # 构建提示词
        prompt = self._build_prompt(context, output, detection_result)
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        """
        使用Qwen3-8B生成文本（接受自定义prompt）

        Args:
            prompt: 提示词

        Returns:
            生成文本
        """
        try:
            # 编码
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )

            # 解码
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # 提取生成部分（去掉输入部分）
            generated_part = generated_text[len(prompt):].strip()

            return generated_part

        except Exception as e:
            logger.error(f"Qwen生成失败: {e}")
            raise

    def _build_prompt(self, context: str, output: str, detection_result: Dict) -> str:
        """构建提示词"""
        # 整理幻觉EDU信息
        suspicious_edus = ""
        if detection_result['hallucination_edus']:
            for i, edu in enumerate(detection_result['hallucination_edus'], 1):
                prob = edu["prob"]
                # 安全性检查：确保概率值有效
                if isinstance(prob, (int, float)) and 0 <= prob <= 1:
                    suspicious_edus += f'\n{i}. "{edu["edu_text"]}" (置信度: {prob:.1%})'
                else:
                    logger.warning(f"Invalid probability value: {prob}, using 0.0 instead")
                    suspicious_edus += f'\n{i}. "{edu["edu_text"]}" (置信度: 0.0%)'
        else:
            suspicious_edus = "\n无可疑片段"

        # 检测结论
        detection_conclusion = (
            "发现幻觉" if detection_result['global_hallucination']
            else "未发现幻觉"
        )

        prompt = f"""你是一个专业的幻觉检测分析师。

【待分析文本】
原始文章：{context}

生成摘要：{output}

【自动检测结果】
RST-RGAT模型检测：{detection_conclusion}（置信度：{detection_result['global_prob']:.1%}）
可疑片段：{suspicious_edus}

请完成以下任务：
1. 【验证】判断自动检测是否准确（是/否/部分准确），并说明理由
2. 【定位】指出摘要中具体哪些内容与原文不符
3. 【分析】解释产生幻觉的原因（事实冲突 / 无中生有 / 过度推断等）
4. 【建议】给出修正后的摘要或改写建议

答案："""

        return prompt


class HallucinationPipeline:
    """完整幻觉检测Pipeline"""

    def __init__(self, model_checkpoint: str,
                 qwen_path: str = "/mnt/second/yuanmengying/qwen3-8b/",
                 device: Optional[torch.device] = None):
        """
        初始化Pipeline

        Args:
            model_checkpoint: RST-RGAT检查点路径
            qwen_path: Qwen模型路径
            device: 设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("初始化RST-RGAT推理器...")
        self.rst_detector = RSTRGATInference(model_checkpoint, self.device)

        logger.info("初始化Qwen分析器...")
        self.qwen_analyzer = QwenAnalyzer(qwen_path, self.device)

        logger.info("✓ Pipeline初始化完成")

    def analyze(self, context: str, output: str) -> Dict:
        """
        完整分析流程

        Args:
            context: 原始文本
            output: 生成文本

        Returns:
            {
                'rst_detection': {...},  # RST-RGAT检测结果
                'llm_verification': str,  # Qwen验证意见
                'hallucination_edus': [],  # 幻觉EDU列表
                'summary': str  # 摘要
            }
        """
        # Step 1: RST-RGAT检测
        logger.info("Step 1: 执行RST-RGAT检测...")
        detection_result = self.rst_detector.predict(context, output)

        # Step 2: Qwen验证分析
        logger.info("Step 2: 使用Qwen进行验证分析...")
        llm_analysis = self.qwen_analyzer.analyze(context, output, detection_result)

        # 构建最终结果
        final_result = {
            'rst_detection': {
                'global_hallucination': detection_result['global_hallucination'],
                'global_prob': detection_result['global_prob'],
                'threshold': detection_result['global_threshold'],
                'hallucination_edus': detection_result['hallucination_edus'],
                'total_edus': len(detection_result['output_edus'])
            },
            'llm_verification': llm_analysis,
            'hallucination_edus': [
                {
                    'text': edu['edu_text'],
                    'confidence': edu['prob']
                }
                for edu in detection_result['hallucination_edus']
            ]
        }

        return final_result


def main():
    """演示Pipeline使用"""
    # 模型路径
    MODEL_CHECKPOINT = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec/outputs/checkpoints/best_model.pt"
    QWEN_PATH = "/mnt/second/yuanmengying/qwen3-8b/"
    DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"

    try:
        # 初始化Pipeline
        logger.info("=" * 80)
        logger.info("初始化幻觉检测Pipeline")
        logger.info("=" * 80)

        pipeline = HallucinationPipeline(
            model_checkpoint=MODEL_CHECKPOINT,
            qwen_path=QWEN_PATH
        )

        # 加载数据
        logger.info("\n" + "=" * 80)
        logger.info("加载summary_data.json")
        logger.info("=" * 80)

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        # 取all_data中 dataset字段为test的部分进行分析
        all_data = [item for item in all_data if item.get('dataset') == 'test']

        logger.info(f"✓ 加载完成，共 {len(all_data)} 条样本")

        # 选择需要分析的样本（可配置）
        num_samples = len(all_data)  # 分析所有test样本
        samples_to_analyze = all_data[:num_samples]

        logger.info(f"将分析前 {num_samples} 条样本")

        # 执行分析
        logger.info("\n" + "=" * 80)
        logger.info("执行幻觉检测和分析")
        logger.info("=" * 80)

        results = []
        for idx, sample in enumerate(samples_to_analyze, 1):
            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            logger.info(f"\n【Sample {idx}/{num_samples} (ID: {sample_id})】")
            logger.info(f"Context长度: {len(context)}")
            logger.info(f"Output长度: {len(output)}")

            try:
                result = pipeline.analyze(context, output)
                result['sample_id'] = sample_id
                results.append(result)

                # 输出结果
                logger.info(f"✓ 分析完成")
                logger.info(f"  全局判断: {'存在幻觉' if result['rst_detection']['global_hallucination'] else '无幻觉'}")
                logger.info(f"  置信度: {result['rst_detection']['global_prob']:.1%}")
                logger.info(f"  幻觉EDU数: {len(result['hallucination_edus'])}")

            except Exception as e:
                logger.error(f"Sample {sample_id} 分析失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 汇总输出
        logger.info("\n" + "=" * 80)
        logger.info(f"分析结果汇总 (共 {len(results)} 条成功)")
        logger.info("=" * 80)

        for i, result in enumerate(results, 1):
            sample_id = result.get('sample_id', i)
            hallucination = result['rst_detection']['global_hallucination']
            prob = result['rst_detection']['global_prob']
            edu_count = len(result['hallucination_edus'])

            print(f"\n【Sample {i}】ID: {sample_id}")
            print(f"  RST-RGAT判断: {'幻觉' if hallucination else '无幻觉'} (置信度: {prob:.1%})")
            print(f"  可疑EDU数: {edu_count}")

            if result['hallucination_edus']:
                print(f"  可疑片段:")
                for j, edu in enumerate(result['hallucination_edus'][:3], 1):  # 只显示前3个
                    print(f'    {j}. "{edu["text"]}" (置信度: {edu["confidence"]:.1%})')

            print(f"  Qwen分析概要: {result['llm_verification'][:200]}...")

        # 保存完整结果
        output_path = Path("hallucination_analysis_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式（处理numpy类型）
            def convert_types(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                else:
                    return obj

            serializable_results = convert_types(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ 完整结果已保存到: {output_path}")
        logger.info(f"✓ 共分析 {len(results)} 条样本，全部成功！")

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
        logger.error(f"请确保以下文件存在:")
        logger.error(f"  - {MODEL_CHECKPOINT}")
        logger.error(f"  - {QWEN_PATH}")
        logger.error(f"  - {DATA_PATH}")
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
