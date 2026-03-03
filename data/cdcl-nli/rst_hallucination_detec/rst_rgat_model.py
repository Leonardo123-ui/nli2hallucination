"""
RST-RGAT模型架构
功能：
1. 关系感知图注意力层（RGAT）
2. 跨图交互注意力（Cross-Attention）
3. 双头输出（EDU级 + 全局级）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RGATLayer(MessagePassing):
    """
    关系感知图注意力层
    支持多种边类型的注意力计算
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 20,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__(aggr='add', node_dim=0)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim必须能被num_heads整除"

        # 每种关系类型的变换矩阵
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])

        # 注意力参数（每种关系一个，每个头一个向量）
        self.att = nn.ParameterList([
            nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
            for _ in range(num_relations)
        ])

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 残差连接的投影
        self.W_skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E,) 边类型ID

        Returns:
            (N, out_dim) 更新后的节点特征
        """
        # 保存原始特征用于残差
        x_skip = x

        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 残差连接
        if self.W_skip is not None:
            x_skip = self.W_skip(x_skip)

        out = out + x_skip

        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor, index: torch.Tensor,
                size_i: int) -> torch.Tensor:
        """
        计算消息（根据边类型）

        Args:
            x_i: (E, in_dim) 目标节点特征
            x_j: (E, in_dim) 源节点特征
            edge_attr: (E,) 边类型
            index: (E,) 目标节点索引
            size_i: 目标节点总数

        Returns:
            (E, out_dim) 消息
        """
        output = torch.zeros(x_i.size(0), self.out_dim, device=x_i.device)

        # 按关系类型分组处理
        for r in range(self.num_relations):
            mask = (edge_attr == r)
            if not mask.any():
                continue

            n_r = mask.sum().item()
            dst_idx = index[mask]

            # 变换特征
            h_i = self.W_r[r](x_i[mask])  # (E_r, out_dim)
            h_j = self.W_r[r](x_j[mask])

            # 重塑为多头 (E_r, num_heads, head_dim)
            h_i_mh = h_i.view(n_r, self.num_heads, self.head_dim)
            h_j_mh = h_j.view(n_r, self.num_heads, self.head_dim)

            # 拼接用于注意力计算
            h_cat = torch.cat([h_i_mh, h_j_mh], dim=-1)  # (E_r, num_heads, 2*head_dim)

            # 计算注意力分数
            # att[r]: (num_heads, 2*head_dim) -> unsqueeze(0) -> (1, num_heads, 2*head_dim)
            e = self.leaky_relu(
                (h_cat * self.att[r].unsqueeze(0)).sum(dim=-1)
            )  # (E_r, num_heads)

            # Per-head softmax归一化（按目标节点分组）
            alpha_list = []
            for h in range(self.num_heads):
                alpha_h = softmax(e[:, h], dst_idx, num_nodes=size_i)
                alpha_list.append(alpha_h)
            alpha = torch.stack(alpha_list, dim=1)  # (E_r, num_heads)
            alpha = self.dropout(alpha)

            # 应用注意力权重
            msg = (h_j_mh * alpha.unsqueeze(-1)).reshape(n_r, self.out_dim)
            output[mask] = msg

        return output


class CrossGraphAttention(nn.Module):
    """
    跨图交互注意力
    让摘要节点从原文节点中提取信息
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, output_features: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_features: (N_out, hidden_dim) 摘要节点特征
            context_features: (N_ctx, hidden_dim) 原文节点特征

        Returns:
            (N_out, hidden_dim) 增强后的摘要节点特征
        """
        N_out = output_features.size(0)
        N_ctx = context_features.size(0)

        # 计算Q, K, V
        Q = self.W_q(output_features).view(N_out, self.num_heads, self.head_dim)
        K = self.W_k(context_features).view(N_ctx, self.num_heads, self.head_dim)
        V = self.W_v(context_features).view(N_ctx, self.num_heads, self.head_dim)

        # 计算注意力分数 (N_out, num_heads, N_ctx)
        attn = torch.einsum('qhd,khd->qhk', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力权重
        out = torch.einsum('qhk,khd->qhd', attn, V)
        out = out.reshape(N_out, -1)

        return out


class RSTRGATModel(nn.Module):
    """
    RST-RGAT主模型
    包含RGAT层、跨图注意力、双头输出
    """

    def __init__(self, input_dim: int = 1027, hidden_dim: int = 512,
                 num_rgat_layers: int = 2, num_relations: int = 20,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rgat_layers = num_rgat_layers

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # RGAT层堆叠
        self.rgat_layers = nn.ModuleList([
            RGATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_relations=num_relations,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_rgat_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_rgat_layers)
        ])

        # 跨图注意力
        self.cross_attention = CrossGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # EDU级分类头
        self.edu_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 全局分类头
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"✓ RST-RGAT模型创建完成")
        logger.info(f"  输入维度: {input_dim}, 隐藏维度: {hidden_dim}")
        logger.info(f"  RGAT层数: {num_rgat_layers}, 关系类型数: {num_relations}")

    def forward(self, data) -> dict:
        """
        Args:
            data: PyG Batch对象，包含x, edge_index, edge_attr, num_context, num_output

        Returns:
            {
                'edu_logits': (total_output_edus,) EDU级别预测（所有图的输出EDU）
                'global_logit': (batch_size,) 全局级别预测（每个图一个）
            }
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 1. 输入投影
        h = self.input_proj(x)

        # 2. RGAT层传播（在整个批次图上进行）
        for rgat, norm in zip(self.rgat_layers, self.layer_norms):
            h = rgat(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)

        # 3. 处理批次中的每个图
        # 获取每个图的num_context和num_output
        num_contexts = data.num_context  # (batch_size,) tensor
        num_outputs = data.num_output    # (batch_size,) tensor

        # Handle both batched and single-graph cases
        if not isinstance(num_contexts, torch.Tensor):
            num_contexts = torch.tensor([num_contexts], device=x.device)
        if not isinstance(num_outputs, torch.Tensor):
            num_outputs = torch.tensor([num_outputs], device=x.device)

        batch_size = num_contexts.size(0)

        all_edu_logits = []
        all_global_logits = []

        node_offset = 0
        for i in range(batch_size):
            n_ctx = num_contexts[i].item()
            n_out = num_outputs[i].item()

            # 提取当前图的context和output节点特征
            h_context = h[node_offset:node_offset + n_ctx]
            h_output = h[node_offset + n_ctx:node_offset + n_ctx + n_out]

            # 跨图注意力增强
            h_output_enhanced = self.cross_attention(h_output, h_context)

            # 拼接原始和增强特征
            h_output_combined = torch.cat([h_output, h_output_enhanced], dim=-1)

            # EDU级预测
            edu_logits = self.edu_classifier(h_output_combined).squeeze(-1)
            all_edu_logits.append(edu_logits)

            # 全局预测（Max-Pooling）
            max_pooled = h_output_combined.max(dim=0, keepdim=True)[0]
            global_logit = self.global_classifier(max_pooled).squeeze()
            all_global_logits.append(global_logit)

            node_offset += n_ctx + n_out

        # 合并结果
        edu_logits = torch.cat(all_edu_logits)  # (total_output_edus,)
        global_logits = torch.stack(all_global_logits)  # (batch_size,)

        return {
            'edu_logits': edu_logits,
            'global_logit': global_logits
        }


if __name__ == "__main__":
    # 测试模型
    from torch_geometric.data import Data

    logger.info("开始测试RST-RGAT模型...")

    # 创建模拟数据
    num_context = 10
    num_output = 5
    num_nodes = num_context + num_output

    # 节点特征 (1024 semantic + 1 position + 1 role + 1 isolated = 1027)
    x = torch.randn(num_nodes, 1027)

    # 边（混合RST边和对齐边）
    edge_index = torch.tensor([
        [0, 1, 2, 10, 11, 10, 11],  # src
        [1, 2, 3, 0, 1, 1, 2]       # dst
    ], dtype=torch.long)

    edge_attr = torch.tensor([5, 6, 5, 19, 19, 19, 19], dtype=torch.long)

    # 标签
    y_local = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    y_global = torch.tensor([1.0])

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_local=y_local,
        y_global=y_global,
        num_context=num_context,
        num_output=num_output
    )

    # 创建模型
    model = RSTRGATModel(
        input_dim=1027,
        hidden_dim=512,
        num_rgat_layers=2,
        num_relations=20,
        num_heads=4
    )

    # 前向传播
    with torch.no_grad():
        outputs = model(data)

    logger.info(f"\n模型输出:")
    logger.info(f"  EDU logits shape: {outputs['edu_logits'].shape}")
    logger.info(f"  EDU logits: {outputs['edu_logits']}")
    logger.info(f"  Global logit: {outputs['global_logit'].item():.4f}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n  模型参数量: {total_params:,}")

    logger.info("\n✓ RST-RGAT模型测试完成！")
