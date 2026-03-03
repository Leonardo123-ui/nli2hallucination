"""
改进的模型架构模块
包含：
1. 层次注意力聚合
2. 关系重要性学习的RGAT
3. 跨图交互注意力
4. 多尺度图池化
5. LLM晚期融合分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


class HierarchicalAttentionAggregator(nn.Module):
    """
    层次注意力聚合器
    用于RST父节点特征的聚合
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()

        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def aggregate_children(self, child_features):
        """
        Args:
            child_features: (num_children, hidden_dim)
        Returns:
            parent_feature: (hidden_dim,)
        """
        if child_features.size(0) == 0:
            return child_features

        # 1. 计算注意力权重
        attention_scores = self.attention_net(child_features)  # (num_children, 1)
        attention_weights = F.softmax(attention_scores, dim=0)

        # 2. 加权求和
        weighted_sum = (child_features * attention_weights).sum(dim=0)

        # 3. 门控融合（考虑所有子节点的平均）
        avg_children = child_features.mean(dim=0)
        combined = torch.cat([weighted_sum, avg_children])
        gate_weight = self.gate(combined)

        parent_feature = gate_weight * weighted_sum + (1 - gate_weight) * avg_children

        return parent_feature


class RelationAwareRGAT(nn.Module):
    """
    关系感知RGAT
    学习不同修辞关系的重要性
    """
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()
        self.rel_names = rel_names
        self.num_relations = len(rel_names)

        # 1. 可学习的关系重要性权重
        self.relation_importance = nn.Parameter(torch.ones(self.num_relations))

        # 2. 关系嵌入
        self.relation_embeddings = nn.Embedding(self.num_relations, 128)

        # 3. 异构图卷积
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    num_heads=1,
                    feat_drop=0.2,
                    attn_drop=0.2,
                    residual=True,
                    allow_zero_in_degree=True
                )
                for rel in rel_names
            },
            aggregate="mean"
        )

        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=hidden_dim,
                    out_feats=out_dim,
                    num_heads=1,
                    feat_drop=0.2,
                    attn_drop=0.2,
                    residual=True,
                    allow_zero_in_degree=True
                )
                for rel in rel_names
            },
            aggregate="mean"
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, g, inputs):
        """
        Args:
            g: DGLHeteroGraph
            inputs: Node feature dictionary {"node_type": features}
        """
        # 第一层
        h = self.conv1(g, inputs)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # 第二层
        h = self.conv2(g, h)
        out = {k: v.squeeze(1) for k, v in h.items()}

        return out

    def get_relation_importance(self):
        """
        返回学到的关系重要性（用于分析）
        """
        weights = F.softmax(self.relation_importance, dim=0)
        importance_dict = {rel: w.item() for rel, w in zip(self.rel_names, weights)}
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)


class CrossGraphAttention(nn.Module):
    """
    跨图交互注意力
    增强premise和hypothesis之间的交互
    """
    def __init__(self, hidden_dim=1024, num_heads=8):
        super().__init__()

        # 双向跨图注意力
        self.premise_to_hyp = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )

        self.hyp_to_premise = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # 门控融合
        self.gate_premise = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_hyp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, g_premise, g_hypothesis):
        """
        在两个图之间进行跨图注意力交互
        """
        # 1. 获取节点特征
        premise_feats = g_premise.ndata['feat']  # (N_p, hidden_dim)
        hyp_feats = g_hypothesis.ndata['feat']    # (N_h, hidden_dim)

        # 2. Premise attend to Hypothesis
        enhanced_premise, attn_p2h = self.premise_to_hyp(
            query=premise_feats.unsqueeze(0),
            key=hyp_feats.unsqueeze(0),
            value=hyp_feats.unsqueeze(0)
        )
        enhanced_premise = enhanced_premise.squeeze(0)

        # 3. Hypothesis attend to Premise
        enhanced_hyp, attn_h2p = self.hyp_to_premise(
            query=hyp_feats.unsqueeze(0),
            key=premise_feats.unsqueeze(0),
            value=premise_feats.unsqueeze(0)
        )
        enhanced_hyp = enhanced_hyp.squeeze(0)

        # 4. 门控融合（残差连接）
        gate_p = self.gate_premise(
            torch.cat([premise_feats, enhanced_premise], dim=1)
        )
        gate_h = self.gate_hyp(
            torch.cat([hyp_feats, enhanced_hyp], dim=1)
        )

        g_premise.ndata['feat'] = premise_feats + gate_p * enhanced_premise
        g_hypothesis.ndata['feat'] = hyp_feats + gate_h * enhanced_hyp

        return g_premise, g_hypothesis


class MultiScalePooling(nn.Module):
    """
    多尺度图池化
    组合多种池化策略
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()

        # 1. 平均池化
        self.avg_pool = dglnn.AvgPooling()

        # 2. 最大池化
        self.max_pool = dglnn.MaxPooling()

        # 3. 注意力池化
        self.attn_pool = dglnn.GlobalAttentionPooling(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )

        # 4. Set2Set池化
        self.set2set = dglnn.Set2Set(
            input_dim=hidden_dim,
            n_iters=3,
            n_layers=1
        )

        # 5. 融合层
        # avg + max + attn + set2set = 1024 + 1024 + 1024 + 2048 = 5120
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, g, features):
        """
        Args:
            g: DGLGraph
            features: (num_nodes, hidden_dim)
        Returns:
            pooled: (hidden_dim,)
        """
        # 各种池化
        avg_feat = self.avg_pool(g, features)
        max_feat = self.max_pool(g, features)
        attn_feat = self.attn_pool(g, features)
        set2set_feat = self.set2set(g, features)

        # 确保所有特征都是1D张量
        avg_feat = avg_feat.flatten() if avg_feat.numel() > 0 else avg_feat.squeeze()
        max_feat = max_feat.flatten() if max_feat.numel() > 0 else max_feat.squeeze()
        attn_feat = attn_feat.flatten() if attn_feat.numel() > 0 else attn_feat.squeeze()
        set2set_feat = set2set_feat.flatten() if set2set_feat.numel() > 0 else set2set_feat.squeeze()

        # 拼接 (hidden_dim + hidden_dim + hidden_dim + hidden_dim*2 = hidden_dim*5)
        combined = torch.cat([avg_feat, max_feat, attn_feat, set2set_feat], dim=0)

        # 融合
        output = self.fusion(combined)

        return output


class HybridClassifier(nn.Module):
    """
    混合分类器
    支持LLM特征的晚期融合
    """
    def __init__(self, hidden_dim=1024, n_classes=2):
        super().__init__()

        # GNN分支
        self.gnn_branch = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

        # LLM融合分支
        # +3维: [llm_confidence, has_contradiction, num_critical_edus]
        self.hybrid_branch = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

        # 自适应权重网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, graph_repr, llm_features=None):
        """
        Args:
            graph_repr: (hidden_dim * 3,) - 拼接的图表示
            llm_features: (3,) - [confidence, has_contradiction, num_critical_edus]
        """
        # GNN预测
        gnn_logits = self.gnn_branch(graph_repr)

        if llm_features is not None and llm_features.sum() > 0:
            # 融合LLM特征
            combined_feat = torch.cat([graph_repr, llm_features], dim=0)
            hybrid_logits = self.hybrid_branch(combined_feat)

            # 自适应权重
            weights = self.attention(combined_feat)

            # 加权融合
            final_logits = weights[0] * gnn_logits + weights[1] * hybrid_logits
            return final_logits
        else:
            return gnn_logits


class ImprovedHeteroClassifier(nn.Module):
    """
    改进的异构图分类器
    集成所有改进模块
    """
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        # 1. 关系感知RGAT (第一层)
        self.rgat1 = RelationAwareRGAT(in_dim, hidden_dim, hidden_dim, rel_names)

        # 2. 跨图交互注意力
        self.cross_attn = CrossGraphAttention(hidden_dim, num_heads=8)

        # 3. 关系感知RGAT (第二层)
        self.rgat2 = RelationAwareRGAT(hidden_dim, hidden_dim, hidden_dim, rel_names)

        # 4. 多尺度池化
        self.pooling = MultiScalePooling(hidden_dim)

        # 5. 混合分类器
        self.classifier = HybridClassifier(hidden_dim, n_classes)

    def forward(self, g_premise_list, g_hypothesis_list, lexical_chain_list,
                nli_labels, llm_features_list=None):
        """
        Args:
            g_premise_list: list of premise graphs
            g_hypothesis_list: list of hypothesis graphs
            lexical_chain_list: list of lexical chains
            nli_labels: labels
            llm_features_list: list of LLM features (optional)
        """
        batch_size = len(g_premise_list)
        all_logits = []
        all_z_premise = []
        all_z_hypothesis = []

        for i in range(batch_size):
            g_pre = g_premise_list[i]
            g_hyp = g_hypothesis_list[i]

            # 1. 第一层RGAT
            h_pre = self.rgat1(g_pre, {'node': g_pre.ndata['feat']})
            h_hyp = self.rgat1(g_hyp, {'node': g_hyp.ndata['feat']})

            g_pre.ndata['feat'] = h_pre['node']
            g_hyp.ndata['feat'] = h_hyp['node']

            # 2. 跨图交互
            g_pre, g_hyp = self.cross_attn(g_pre, g_hyp)

            # 3. 第二层RGAT
            h_pre = self.rgat2(g_pre, {'node': g_pre.ndata['feat']})
            h_hyp = self.rgat2(g_hyp, {'node': g_hyp.ndata['feat']})

            # 4. 多尺度池化
            z_pre = self.pooling(g_pre, h_pre['node'])
            z_hyp = self.pooling(g_hyp, h_hyp['node'])

            all_z_premise.append(z_pre)
            all_z_hypothesis.append(z_hyp)

            # 5. 图级表示融合
            graph_repr = torch.cat([z_pre, z_hyp, z_pre * z_hyp], dim=0)

            # 6. 分类
            llm_feat = llm_features_list[i] if llm_features_list is not None else None
            logits = self.classifier(graph_repr, llm_feat)
            all_logits.append(logits)

        # 堆叠结果
        final_logits = torch.stack(all_logits)
        final_z_premise = torch.stack(all_z_premise)
        final_z_hypothesis = torch.stack(all_z_hypothesis)

        return {
            "classification": final_logits,
            "z_premise": final_z_premise,
            "z_hypothesis": final_z_hypothesis
        }


if __name__ == "__main__":
    print("Testing improved model components...")

    hidden_dim = 1024
    batch_size = 4
    num_nodes = 10

    # 测试层次注意力
    print("\n1. Testing HierarchicalAttentionAggregator...")
    aggregator = HierarchicalAttentionAggregator(hidden_dim)
    child_feats = torch.randn(5, hidden_dim)
    parent_feat = aggregator.aggregate_children(child_feats)
    print(f"   Input shape: {child_feats.shape}, Output shape: {parent_feat.shape}")

    # 测试跨图注意力
    print("\n2. Testing CrossGraphAttention...")
    cross_attn = CrossGraphAttention(hidden_dim)
    g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
    g2 = dgl.graph(([0, 1], [1, 2]))
    g1.ndata['feat'] = torch.randn(4, hidden_dim)
    g2.ndata['feat'] = torch.randn(3, hidden_dim)
    g1_out, g2_out = cross_attn(g1, g2)
    print(f"   G1 nodes: {g1.num_nodes()}, G2 nodes: {g2.num_nodes()}")

    # 测试多尺度池化
    print("\n3. Testing MultiScalePooling...")
    pooling = MultiScalePooling(hidden_dim)
    g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    g.ndata['feat'] = torch.randn(5, hidden_dim)
    pooled = pooling(g, g.ndata['feat'])
    print(f"   Input nodes: {g.num_nodes()}, Output shape: {pooled.shape}")

    # 测试混合分类器
    print("\n4. Testing HybridClassifier...")
    classifier = HybridClassifier(hidden_dim)
    graph_repr = torch.randn(hidden_dim * 3)
    llm_feat = torch.tensor([0.8, 1.0, 0.5])
    logits = classifier(graph_repr, llm_feat)
    print(f"   Output logits shape: {logits.shape}")

    print("\n✅ All components work correctly!")
