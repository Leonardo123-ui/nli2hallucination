"""
LLM增强的图构建模块
将LLM的分析结果（Rationale）融入DGL图的节点特征中
"""

import torch
import base64
import re
import logging
from typing import List, Tuple, Dict, Optional
import dgl

logger = logging.getLogger(__name__)


def decode_text_from_encoded(encoded_bytes: bytes) -> str:
    """
    从base64编码的字节中解码文本

    Args:
        encoded_bytes: Base64编码的字节

    Returns:
        解码后的原始文本字符串
    """
    try:
        # 去除填充的空字节
        byte_data = encoded_bytes.rstrip(b"\0")
        # Base64解码
        decoded_text = base64.b64decode(byte_data).decode("utf-8")
        return decoded_text
    except Exception as e:
        logger.warning(f"解码文本失败: {e}")
        return ""


def extract_keywords_from_rationale(rationale: str, top_k: int = 10) -> List[str]:
    """
    从LLM的rationale中提取关键词

    Args:
        rationale: LLM生成的分析文本
        top_k: 提取的关键词数量

    Returns:
        关键词列表
    """
    if not rationale or len(rationale) == 0:
        return []

    # 1. 移除特殊字符和数字，保留中英文和空格
    cleaned = re.sub(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z\s，。！？\-]', '', rationale)

    # 2. 按多种分隔符分词（中文按字、英文按词）
    # 简单方案：中文文本按字分割，英文单词通过空格分割
    words = []

    # 提取英文单词
    english_words = re.findall(r'\b[a-zA-Z]+\b', cleaned, re.IGNORECASE)
    words.extend(english_words)

    # 提取中文词组（长度为2-4的连续中文字符）
    # 这里用简单的滑动窗口，更精准的方法需要分词库（jieba等）
    chinese_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', cleaned))

    # 提取长度为2-4的中文词组
    for i in range(len(chinese_text) - 1):
        for length in [3, 2]:  # 优先提取3字词组，再提取2字词组
            if i + length <= len(chinese_text):
                word = chinese_text[i:i+length]
                if word not in words:  # 避免重复
                    words.append(word)

    # 3. 过滤停用词
    stopwords = {
        '的', '了', '是', '在', '和', '有', '一', '个', '这', '为', '不',
        '人', '我', '他', '她', '它', '们', '可以', '但是', '所以', '因为',
        '然而', '因此', '其他', '另外', '更', '也', '都', '很', '就', '去',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of',
        'is', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    }

    filtered_words = [w for w in words if w.lower() not in stopwords and len(w) > 0]

    # 4. 计算词频，返回top-k
    from collections import Counter
    word_freq = Counter(filtered_words)
    top_keywords = [w for w, _ in word_freq.most_common(top_k)]

    logger.debug(f"从rationale提取的关键词: {top_keywords}")
    return top_keywords


def get_node_text_from_graph(graph: dgl.DGLGraph, node_id: int) -> str:
    """
    从图中获取节点的原始文本

    Args:
        graph: DGL图
        node_id: 节点ID

    Returns:
        节点的原始文本
    """
    try:
        # 尝试从ndata["text_encoded"]中获取编码的文本
        if "text_encoded" in graph.ndata:
            encoded = graph.ndata["text_encoded"][node_id]
            # 转换为字节
            if isinstance(encoded, torch.Tensor):
                byte_data = bytes(encoded.cpu().numpy().astype(int))
            else:
                byte_data = bytes(encoded)
            return decode_text_from_encoded(byte_data)
        else:
            logger.warning(f"图中没有text_encoded属性")
            return ""
    except Exception as e:
        logger.debug(f"获取节点{node_id}文本失败: {e}")
        return ""


def match_keywords_to_nodes(
    graph: dgl.DGLGraph,
    keywords: List[str],
    similarity_threshold: float = 0.5
) -> Dict[int, float]:
    """
    将关键词匹配到图中的节点，计算节点权重

    Args:
        graph: DGL图
        keywords: 关键词列表
        similarity_threshold: 相似度阈值（0.5表示关键词出现在节点文本中的50%以上）

    Returns:
        节点ID到权重的映射 {node_id: weight}
    """
    node_weights = {}
    num_nodes = graph.num_nodes()

    # 初始化所有节点权重为普通权重（1.0 - 不改变特征）
    # 关键节点权重为 1.2（增强20%）
    base_weight = 1.0  # 修改: 从0.5改为1.0,避免削弱普通节点特征
    keyword_weight = 1.2  # 修改: 从1.0改为1.2,适度增强关键节点

    for node_id in range(num_nodes):
        node_text = get_node_text_from_graph(graph, node_id)
        node_text_lower = node_text.lower()

        # 计算节点与关键词的匹配分数
        match_score = 0.0
        keyword_matches = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in node_text_lower:
                keyword_matches += 1
                # 权重：关键词数量越多，权重越高
                match_score = min(1.0, keyword_matches / len(keywords) * 1.5)

        # 根据匹配分数计算最终权重
        if match_score > 0:
            # 线性插值：base_weight + match_score * (keyword_weight - base_weight)
            node_weights[node_id] = base_weight + match_score * (keyword_weight - base_weight)
        else:
            node_weights[node_id] = base_weight

    logger.debug(f"节点权重分布: {[(nid, w) for nid, w in sorted(node_weights.items())[:5]]}")
    return node_weights


def apply_weights_to_features(
    features: torch.Tensor,
    node_weights: Dict[int, float],
    method: str = "multiply"
) -> torch.Tensor:
    """
    将节点权重应用到特征向量

    Args:
        features: 节点特征矩阵 (num_nodes, feat_dim)
        node_weights: 节点权重字典
        method: 权重应用方法
            - "multiply": 乘法融合 features * weight
            - "concat": 拼接权重作为额外维度
            - "gating": 门控融合 features * weight + (1 - weight) * 0

    Returns:
        增强后的特征矩阵
    """
    features = features.clone()

    if method == "multiply":
        # 最简单的方法：直接乘以权重
        for node_id, weight in node_weights.items():
            features[node_id] = features[node_id] * weight

    elif method == "concat":
        # 拼接权重作为额外维度
        weights_tensor = torch.tensor(
            [node_weights.get(i, 0.5) for i in range(features.shape[0])],
            dtype=features.dtype,
            device=features.device
        ).unsqueeze(1)
        features = torch.cat([features, weights_tensor], dim=1)

    elif method == "gating":
        # 门控融合：强调关键节点，削弱普通节点
        for node_id, weight in node_weights.items():
            # gating = weight * x + (1 - weight) * x_scaled
            if weight > 0.75:  # 关键节点
                features[node_id] = features[node_id] * 1.2
            elif weight < 0.6:  # 普通节点
                features[node_id] = features[node_id] * 0.8

    return features


def build_augmented_graph(
    g_premise: dgl.DGLGraph,
    g_hypothesis: dgl.DGLGraph,
    rationale: str,
    method: str = "multiply"
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """
    使用LLM的rationale增强两个图的节点特征

    核心流程：
    1. 从rationale中提取关键词
    2. 将关键词匹配到premise和hypothesis图中的节点
    3. 计算节点权重
    4. 将权重融入节点特征

    Args:
        g_premise: 前提文本的DGL图
        g_hypothesis: 假设文本的DGL图
        rationale: LLM生成的分析文本
        method: 权重融合方法 ("multiply", "concat", "gating")

    Returns:
        增强后的 (g_premise, g_hypothesis) 图对
    """

    # 克隆图以避免修改原始图
    g_premise = g_premise.clone()
    g_hypothesis = g_hypothesis.clone()

    if not rationale or len(rationale.strip()) == 0:
        logger.warning("Rationale为空，使用默认权重")
        # 即使rationale为空，也要保持schema一致，添加默认权重
        # 所有节点权重设为0.5（不增强也不削弱）
        default_weight = 0.5

        # 为premise图添加默认权重（不修改特征）
        g_premise.ndata["augmentation_weight"] = torch.ones(
            g_premise.num_nodes(),
            dtype=torch.float32,
            device=g_premise.device
        ) * default_weight

        # 为hypothesis图添加默认权重（不修改特征）
        g_hypothesis.ndata["augmentation_weight"] = torch.ones(
            g_hypothesis.num_nodes(),
            dtype=torch.float32,
            device=g_hypothesis.device
        ) * default_weight

        return g_premise, g_hypothesis

    # 1. 从rationale中提取关键词
    keywords = extract_keywords_from_rationale(rationale, top_k=10)

    if len(keywords) == 0:
        logger.warning("未能从rationale中提取关键词，使用默认权重")
        # 保持schema一致：添加默认权重
        default_weight = 0.5

        g_premise.ndata["augmentation_weight"] = torch.ones(
            g_premise.num_nodes(),
            dtype=torch.float32,
            device=g_premise.device
        ) * default_weight

        g_hypothesis.ndata["augmentation_weight"] = torch.ones(
            g_hypothesis.num_nodes(),
            dtype=torch.float32,
            device=g_hypothesis.device
        ) * default_weight

        return g_premise, g_hypothesis

    logger.debug(f"提取的关键词: {keywords}")

    # 2. 为premise图匹配关键词
    premise_weights = match_keywords_to_nodes(g_premise, keywords)
    logger.debug(f"Premise节点权重: {len(premise_weights)} 个节点")

    # 3. 为hypothesis图匹配关键词
    hypothesis_weights = match_keywords_to_nodes(g_hypothesis, keywords)
    logger.debug(f"Hypothesis节点权重: {len(hypothesis_weights)} 个节点")

    # 4. 应用权重到特征
    premise_feat = g_premise.ndata["feat"]
    hypothesis_feat = g_hypothesis.ndata["feat"]

    # 增强特征
    augmented_premise_feat = apply_weights_to_features(premise_feat, premise_weights, method=method)
    augmented_hypothesis_feat = apply_weights_to_features(hypothesis_feat, hypothesis_weights, method=method)

    # 5. 更新图的特征
    g_premise.ndata["feat"] = augmented_premise_feat
    g_hypothesis.ndata["feat"] = augmented_hypothesis_feat

    # 保存权重信息供后续分析
    g_premise.ndata["augmentation_weight"] = torch.tensor(
        [premise_weights.get(i, 0.5) for i in range(g_premise.num_nodes())],
        dtype=torch.float32,
        device=g_premise.device
    )
    g_hypothesis.ndata["augmentation_weight"] = torch.tensor(
        [hypothesis_weights.get(i, 0.5) for i in range(g_hypothesis.num_nodes())],
        dtype=torch.float32,
        device=g_hypothesis.device
    )

    logger.debug(f"图增强完成: premise{g_premise.ndata['feat'].shape}, hypothesis{g_hypothesis.ndata['feat'].shape}")

    return g_premise, g_hypothesis


def augment_batch_graphs(
    graph_batch_premise: List[dgl.DGLGraph],
    graph_batch_hypothesis: List[dgl.DGLGraph],
    rationales: List[str],
    method: str = "multiply"
) -> Tuple[List[dgl.DGLGraph], List[dgl.DGLGraph]]:
    """
    增强一整个批次的图

    Args:
        graph_batch_premise: 前提图列表
        graph_batch_hypothesis: 假设图列表
        rationales: rationale文本列表（每个样本对应一个）
        method: 权重融合方法

    Returns:
        增强后的 (premise_list, hypothesis_list)
    """
    augmented_premise = []
    augmented_hypothesis = []

    for g_p, g_h, rat in zip(graph_batch_premise, graph_batch_hypothesis, rationales):
        try:
            g_p_aug, g_h_aug = build_augmented_graph(g_p, g_h, rat, method=method)
            augmented_premise.append(g_p_aug)
            augmented_hypothesis.append(g_h_aug)
        except Exception as e:
            logger.warning(f"增强图失败: {e}，使用原始图+默认权重")
            # 克隆原始图并添加默认权重，保持schema一致
            g_p_clone = g_p.clone()
            g_h_clone = g_h.clone()

            # 添加默认权重
            default_weight = 0.5
            g_p_clone.ndata["augmentation_weight"] = torch.ones(
                g_p_clone.num_nodes(),
                dtype=torch.float32,
                device=g_p_clone.device
            ) * default_weight

            g_h_clone.ndata["augmentation_weight"] = torch.ones(
                g_h_clone.num_nodes(),
                dtype=torch.float32,
                device=g_h_clone.device
            ) * default_weight

            augmented_premise.append(g_p_clone)
            augmented_hypothesis.append(g_h_clone)

    return augmented_premise, augmented_hypothesis
