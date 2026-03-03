"""
LLM-Augmented Hallucination Detector
核心判别器类实现 - 结合LLM理由注入和本地判别模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, Optional
import os


class AugmentedDiscriminator:
    """
    增强型幻觉检测器 - 结合LLM分析和本地判别模型

    流程:
    1. 调用LLM获取理由 (Rationale)
    2. 将理由与摘要和原文拼接
    3. 使用本地判别模型进行推理
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        llm_client: Optional[object] = None,
    ):
        """
        初始化判别器

        Args:
            model_path: HuggingFace模型路径或本地路径
            device: 计算设备 ('cuda' 或 'cpu')
            llm_client: LLM客户端对象（需要有调用方法）
        """
        self.device = device
        self.llm_client = llm_client

        # 加载本地判别模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

        # 获取模型配置
        self.max_length = getattr(self.tokenizer, 'model_max_length', 512)

    def get_llm_rationale(self, source: str, summary: str) -> str:
        """
        调用LLM获取理由 (Rationale)

        Args:
            source: 原文本
            summary: 摘要文本

        Returns:
            LLM生成的分析理由文本
        """
        if self.llm_client is None:
            # 如果没有LLM客户端，返回默认理由
            return "No LLM analysis available."

        system_prompt = (
            "你是一名幻觉侦探。请简要列出 Summary 和 Source 之间的任何不一致之处。"
            "如果没有，请回答 'No inconsistencies found'。请将字数控制在 50 字以内。"
        )

        user_input = f"Source: {source}\n\nSummary: {summary}"

        try:
            # 调用LLM API（这里是占位符实现）
            # 实际使用时需要替换为真实的API调用
            rationale = self.llm_client.call_llm(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=50
            )
            return rationale
        except Exception as e:
            # 异常处理：返回错误消息
            print(f"Warning: LLM call failed - {str(e)}")
            return "LLM analysis failed."

    def format_input(self, rationale: str, summary: str, source: str) -> str:
        """
        格式化输入文本：拼接理由、摘要和原文

        Args:
            rationale: LLM生成的理由
            summary: 摘要文本
            source: 原文本

        Returns:
            格式化后的输入字符串
        """
        # 使用分词器的特殊标记进行拼接
        sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else '</s>'
        cls_token = self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else '[CLS]'

        # 格式: [CLS] Rationale [SEP] Summary [SEP] Source
        formatted = f"{rationale} {sep_token} {summary} {sep_token} {source}"

        return formatted

    def predict(self, source: str, summary: str) -> Tuple[float, int, Dict]:
        """
        预测幻觉概率和标签

        Args:
            source: 原文本
            summary: 摘要文本

        Returns:
            (概率分数, 预测标签, 详细信息字典)
            - 分数: 0-1之间的浮点数
            - 标签: 0 (无幻觉) 或 1 (有幻觉)
            - 详细信息: 包含rationale等信息
        """
        # 阶段 1: 获取LLM理由
        rationale = self.get_llm_rationale(source, summary)

        # 阶段 2: 格式化输入
        formatted_input = self.format_input(rationale, summary, source)

        # 阶段 3: 分词
        inputs = self.tokenizer(
            formatted_input,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # 移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            # 获取最大概率和对应的标签
            score, predicted_label = torch.max(probabilities, dim=-1)

            score = score.item()
            predicted_label = predicted_label.item()

        details = {
            "rationale": rationale,
            "formatted_input_length": len(formatted_input),
            "tokenized_length": inputs['input_ids'].shape[1],
            "probabilities": probabilities[0].cpu().numpy().tolist(),
        }

        return score, predicted_label, details

    def batch_predict(
        self,
        sources: list,
        summaries: list,
        return_details: bool = False
    ) -> list:
        """
        批量预测

        Args:
            sources: 原文本列表
            summaries: 摘要文本列表
            return_details: 是否返回详细信息

        Returns:
            预测结果列表
        """
        results = []
        for source, summary in zip(sources, summaries):
            score, label, details = self.predict(source, summary)
            result = {"score": score, "label": label}
            if return_details:
                result["details"] = details
            results.append(result)

        return results
