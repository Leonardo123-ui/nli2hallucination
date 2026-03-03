"""
异构图构建模块
功能：
1. 节点特征提取（语义 + 位置 + RST角色）
2. 边构建（RST内部边 + 跨图对齐边）
3. PyTorch Geometric Data对象生成
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeFeatureExtractor:
    """节点特征提取器"""

    def __init__(self, model_path: str = "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        # 特征提取强制使用CPU，避免与训练模型竞争GPU显存
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        logger.info(f"✓ Modern-BERT加载完成（CPU模式）: {model_path}")

    def extract_semantic_features(self, texts: List[str]) -> torch.Tensor:
        """
        提取语义特征（Mean Pooling）

        Args:
            texts: EDU文本列表

        Returns:
            (num_edus, hidden_dim) 语义向量
        """
        if not texts:
            return torch.zeros(0, 1024)

        # 批量编码
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # 强制使用CPU，不移动到CUDA

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Mean Pooling (忽略padding)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        mean_pooled = sum_hidden / sum_mask

        return mean_pooled.cpu()

    def compute_position_encoding(self, num_edus: int) -> torch.Tensor:
        """
        相对位置编码: [0, 1/(n-1), 2/(n-1), ..., 1]

        Args:
            num_edus: EDU数量

        Returns:
            (num_edus, 1) 位置特征
        """
        if num_edus == 1:
            return torch.zeros(1, 1)

        positions = torch.arange(num_edus, dtype=torch.float32)
        positions = positions / (num_edus - 1)

        return positions.unsqueeze(1)

    def extract_rst_role(self, tree: str, num_edus: int) -> torch.Tensor:
        """
        提取RST角色（Nucleus=1, Satellite=0）

        Args:
            tree: DMRST树表达式 "(1:Satellite=Contrast:4, 5:Nucleus=span:6)"
            num_edus: EDU数量

        Returns:
            (num_edus, 1) 角色特征
        """
        roles = torch.zeros(num_edus, 1)

        # 解析树结构，标记Nucleus节点
        nucleus_pattern = r'(\d+):Nucleus'
        for match in re.finditer(nucleus_pattern, tree):
            edu_id = int(match.group(1)) - 1  # DMRST从1开始索引
            if 0 <= edu_id < num_edus:
                roles[edu_id] = 1.0

        return roles

    def build_node_features(self, parsed_data: Dict) -> torch.Tensor:
        """
        构建完整节点特征矩阵

        Args:
            parsed_data: {
                'segments': ['EDU1', 'EDU2', ...],
                'tree': '(1:Satellite=...)',
                ...
            }

        Returns:
            (num_edus, feature_dim) 特征矩阵
        """
        segments = parsed_data['segments']
        tree = parsed_data['tree']
        num_edus = len(segments)

        # 1. 语义特征
        semantic_features = self.extract_semantic_features(segments)  # (N, 1024)

        # 2. 位置编码
        position_features = self.compute_position_encoding(num_edus)  # (N, 1)

        # 3. RST角色
        role_features = self.extract_rst_role(tree, num_edus)  # (N, 1)

        # 拼接
        features = torch.cat([
            semantic_features,
            position_features,
            role_features
        ], dim=1)

        return features


class EdgeBuilder:
    """边构建器"""

    # RST关系映射表（简化版，仅保留主要关系）
    RST_RELATIONS = {
        'Attribution': 0, 'Background': 1, 'Cause': 2, 'Comparison': 3,
        'Condition': 4, 'Contrast': 5, 'Elaboration': 6, 'Enablement': 7,
        'Evaluation': 8, 'Explanation': 9, 'Joint': 10, 'Manner-Means': 11,
        'Topic-Comment': 12, 'Summary': 13, 'Temporal': 14, 'Topic-Change': 15,
        'span': 16, 'Same-Unit': 17, 'Textual-Organization': 18
    }
    ALIGNMENT_RELATION = 19  # 跨图对齐关系

    @staticmethod
    def parse_rst_tree(tree: str, num_edus: int) -> Tuple[List, List]:
        """
        解析DMRST树为边列表

        Args:
            tree: "(1:Satellite=Contrast:4, 5:Nucleus=span:6)"
            num_edus: EDU数量

        Returns:
            (edge_index, edge_type)
            edge_index: [[src1, dst1], [src2, dst2], ...]
            edge_type: [rel_id1, rel_id2, ...]
        """
        edges = []
        edge_types = []

        # 解析括号表达式
        pattern = r'(\d+):(Satellite|Nucleus)=(\w+[-\w]*):(\d+)'
        matches = re.findall(pattern, tree)

        for match in matches:
            start_edu = int(match[0]) - 1
            role = match[1]
            relation = match[2]
            end_edu = int(match[3]) - 1

            # 范围检查
            if not (0 <= start_edu < num_edus and 0 <= end_edu < num_edus):
                continue

            # 获取关系ID
            rel_id = EdgeBuilder.RST_RELATIONS.get(relation, 16)

            # Satellite -> Nucleus: 单向边
            if role == 'Satellite':
                edges.append([start_edu, end_edu])
                edge_types.append(rel_id)

            # Nucleus-Nucleus (如Joint): 双向边
            elif relation in ['Joint', 'Same-Unit']:
                edges.append([start_edu, end_edu])
                edges.append([end_edu, start_edu])
                edge_types.extend([rel_id, rel_id])

        return edges, edge_types

    @staticmethod
    def build_cross_alignment_edges(
        context_features: torch.Tensor,
        output_features: torch.Tensor,
        num_context: int,
        num_output: int,
        top_k: int = 2,
        threshold: float = 0.4
    ) -> Tuple[List, List, List]:
        """
        构建跨图对齐边（基于余弦相似度）

        Args:
            context_features: (N_ctx, dim) 原文特征
            output_features: (N_out, dim) 摘要特征
            num_context: 原文EDU数量
            num_output: 摘要EDU数量
            top_k: 每个摘要节点连接的原文节点数
            threshold: 相似度阈值

        Returns:
            (edge_index, edge_type, isolated_flags)
            isolated_flags: 标记孤立节点（无对齐边）
        """
        edges = []
        edge_types = []
        isolated_flags = torch.zeros(num_output)

        # 计算相似度矩阵 (N_out, N_ctx)
        similarity = F.cosine_similarity(
            output_features.unsqueeze(1),  # (N_out, 1, dim)
            context_features.unsqueeze(0),  # (1, N_ctx, dim)
            dim=2
        )

        # 为每个摘要节点选择Top-K原文节点
        for i in range(num_output):
            sims = similarity[i]
            top_values, top_indices = torch.topk(sims, min(top_k, num_context))

            has_edge = False
            for sim_val, ctx_idx in zip(top_values, top_indices):
                if sim_val >= threshold:
                    # 摘要节点 -> 原文节点
                    edges.append([num_context + i, ctx_idx.item()])
                    edge_types.append(EdgeBuilder.ALIGNMENT_RELATION)
                    has_edge = True

            # 标记孤立节点
            if not has_edge:
                isolated_flags[i] = 1.0

        return edges, edge_types, isolated_flags


class GraphBuilder:
    """异构图构建主类"""

    def __init__(self, model_path: str = "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large"):
        self.feature_extractor = NodeFeatureExtractor(model_path)
        self.edge_builder = EdgeBuilder()

    def build_graph(self, processed_sample: Dict) -> Data:
        """
        构建PyTorch Geometric Data对象

        Args:
            processed_sample: {
                'id': '276',
                'context': {'segments': [...], 'tree': '...', ...},
                'output': {'segments': [...], 'tree': '...', ...},
                'output_edu_labels': [0, 1, 1, 0],
                'y_global': 1
            }

        Returns:
            Data对象，包含x, edge_index, edge_attr, y_local, y_global
        """
        context_data = processed_sample['context']
        output_data = processed_sample['output']

        num_context = len(context_data['segments'])
        num_output = len(output_data['segments'])

        # 1. 节点特征提取
        context_features = self.feature_extractor.build_node_features(context_data)
        output_features = self.feature_extractor.build_node_features(output_data)

        # 2. 构建RST内部边
        ctx_edges, ctx_types = self.edge_builder.parse_rst_tree(
            context_data['tree'], num_context
        )
        out_edges, out_types = self.edge_builder.parse_rst_tree(
            output_data['tree'], num_output
        )

        # 调整output边索引（偏移num_context）
        out_edges = [[src + num_context, dst + num_context]
                     for src, dst in out_edges]

        # 3. 构建跨图对齐边
        align_edges, align_types, isolated_flags = self.edge_builder.build_cross_alignment_edges(
            context_features, output_features,
            num_context, num_output
        )

        # 4. 合并所有边
        all_edges = ctx_edges + out_edges + align_edges
        all_types = ctx_types + out_types + align_types

        # 5. 添加孤立节点标记到output特征
        isolated_flags = isolated_flags.unsqueeze(1)
        output_features = torch.cat([output_features, isolated_flags], dim=1)

        # 补齐context特征（添加全0的isolated列）
        context_features = torch.cat([
            context_features,
            torch.zeros(num_context, 1)
        ], dim=1)

        # 6. 合并节点特征
        x = torch.cat([context_features, output_features], dim=0)

        # 7. 转换为Tensor
        if len(all_edges) > 0:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(all_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)

        # 8. 标签（仅output节点有标签）
        y_local = torch.tensor(processed_sample['output_edu_labels'], dtype=torch.float32)
        y_global = torch.tensor([processed_sample['y_global']], dtype=torch.float32)

        # 9. 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_local=y_local,
            y_global=y_global,
            num_context=torch.tensor([num_context], dtype=torch.long),
            num_output=torch.tensor([num_output], dtype=torch.long)
        )

        return data


if __name__ == "__main__":
    # 测试图构建
    logger.info("开始测试图构建模块...")

    # 模拟处理后的样本
    sample = {
        'id': 'test_001',
        'context': {
            'segments': ['Context EDU 1.', 'Context EDU 2.', 'Context EDU 3.'],
            'tree': '(1:Satellite=Background:2, 3:Nucleus=span:3)',
            'edu_breaks': [5, 10],
            'token_offsets': [(0, 15), (16, 31), (32, 47)]
        },
        'output': {
            'segments': ['Output EDU 1.', 'Output EDU 2.'],
            'tree': '(1:Nucleus=Joint:2)',
            'edu_breaks': [5],
            'token_offsets': [(0, 14), (15, 29)]
        },
        'output_edu_labels': [0, 1],
        'y_global': 1
    }

    # 构建图
    builder = GraphBuilder()
    graph_data = builder.build_graph(sample)

    logger.info(f"\n图构建完成:")
    logger.info(f"  总节点数: {graph_data.num_nodes}")
    logger.info(f"  原文节点: {graph_data.num_context}")
    logger.info(f"  摘要节点: {graph_data.num_output}")
    logger.info(f"  边数: {graph_data.num_edges}")
    logger.info(f"  节点特征维度: {graph_data.x.shape}")
    logger.info(f"  局部标签: {graph_data.y_local}")
    logger.info(f"  全局标签: {graph_data.y_global}")
    logger.info(f"  边类型统计: {torch.bincount(graph_data.edge_attr)}")

    logger.info("\n✓ 图构建模块测试完成！")
