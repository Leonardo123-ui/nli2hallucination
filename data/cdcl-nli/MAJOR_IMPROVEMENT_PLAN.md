# 重大结构性改进方案

**分析日期**: 2025-12-27
**当前最优**: Model 3 with Macro F1 = 0.5611
**目标**: 达到 0.58-0.60+

---

## 一、问题诊断

### 为什么微调超参数效果不好？

| Model | Alpha | Gamma | Macro F1 | 变化 |
|-------|-------|-------|----------|------|
| Model 1 | [0.5, 2.0] | 2.0 | 0.5494 | baseline |
| Model 3 | [1.0, 3.4] | 2.0 | 0.5611 | +0.0117 |
| Model 4 | [1.0, 3.4] | 2.5 | 0.5370 | -0.0241 ❌ |

**关键观察**: Model 4 反而下降，说明超参数微调已经触及边界，需要**架构级改动**

---

## 二、三个主要改进方向

### 方向 1️⃣: 关系类型优化 (优先级: ⭐⭐⭐⭐⭐)

#### 当前状况
- 模型使用 **20 个关系类型** (`rel_names_long`)
  - Temporal, TextualOrganization, Joint, Topic-Comment, Comparison, Condition, Contrast, Evaluation, Topic-Change, Summary, Manner-Means, Attribution, Cause, Background, Enablement, Explanation, Same-Unit, Elaboration, span, lexical

- 代码中定义了 **9 个关键关系** (`rel_names_short`)
  - Temporal, Summary, Condition, Contrast, Cause, Background, Elaboration, Explanation, lexical

#### 问题分析
- **数据稀疏**: 20 个关系类型对于 900 个样本可能过多
- **关系噪声**: 某些关系（如 span）可能对分类没有帮助
- **计算效率**: 每个关系都需要单独的 GAT 卷积，增加模型复杂度

#### 改进方案

**方案 1A: 减少到关键的 9 个关系 (推荐)**
```python
# 在 train.py 第 718 行，改为:
model = ExplainableHeteroClassifier(
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    rel_names=rel_names_short,  # ← 改成 short！
)

# 同时在第 496 行：
# model.merge_graphs(g_p1, g_p2, lc, rel_names_short)  # ← 改成 short！
```

**预期效果**:
- 减少模型参数约 55% (20→9 个关系)
- 降低过拟合风险
- 预期 Macro F1: +0.01-0.03

**实施难度**: ⭐ (仅需改两行)

---

**方案 1B: 自适应关系选择 (高级)**

创建一个关系重要性评分机制，只保留对分类有贡献的关系：

```python
class AdaptiveRelationSelector(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        # 学习每个关系的重要性分数
        self.relation_importance = nn.Parameter(torch.ones(num_relations))

    def forward(self, rel_features):
        # 软选择：根据重要性加权聚合
        importance_weights = F.softmax(self.relation_importance, dim=0)
        # 只保留权重 > threshold 的关系
        threshold = 1.0 / len(self.relation_importance)
        selected = importance_weights > threshold
        return selected, importance_weights
```

**预期效果**: +0.02-0.04
**实施难度**: ⭐⭐⭐ (需要修改模型)

---

### 方向 2️⃣: 网络架构增强 (优先级: ⭐⭐⭐⭐)

#### 当前架构分析

```
输入特征 (in_dim=768)
    ↓
RGAT Layer 1 (4 个注意力头, hidden_dim=256)
    ↓
RGAT Layer 2 (1 个注意力头, out_dim=256)
    ↓
分类器 (input=hidden_dim*3=768, output=2 classes)
```

**问题**:
- 只有 2 层卷积，模型容量可能不足
- 第二层只有 1 个注意力头，信息损失
- 没有充分利用多头注意力的优势

#### 改进方案

**方案 2A: 增加网络深度 (推荐)**

在 `build_base_graph_extract.py` 中修改 RGAT 类：

```python
class RGAT_Enhanced(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()
        self.rel_names = rel_names

        # 第一层：高维映射
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(
                in_feats=in_dim,
                out_feats=hidden_dim,
                num_heads=4,
                feat_drop=0.1,
                attn_drop=0.1,
                residual=True,
                allow_zero_in_degree=True,
            )
            for rel in rel_names
        }, aggregate="mean")

        # 第二层：特征精化（新增）
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(
                in_feats=hidden_dim * 4,
                out_feats=hidden_dim,
                num_heads=2,  # ← 改成 2（而不是 1）
                feat_drop=0.1,
                attn_drop=0.1,
                residual=True,
                allow_zero_in_degree=True,
            )
            for rel in rel_names
        }, aggregate="mean")

        # 第三层：深度特征聚合（新增）
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(
                in_feats=hidden_dim * 2,
                out_feats=out_dim,
                num_heads=1,
                feat_drop=0.1,
                attn_drop=0.1,
                residual=True,
                allow_zero_in_degree=True,
            )
            for rel in rel_names
        }, aggregate="mean")

        self.dropout = nn.Dropout(0.15)  # 增加 dropout

    def forward(self, g, inputs, return_attention=False):
        h_dict = inputs

        # 第一层
        h = self.conv1(g, h_dict)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # 第二层（新增）
        h = self.conv2(g, h)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # 第三层（新增）
        h = self.conv3(g, h)
        out = {k: v.squeeze(1) for k, v in h.items()}

        return out
```

**预期效果**: +0.02-0.04
**实施难度**: ⭐⭐⭐ (需要修改 RGAT 类，改变前向传播)

---

**方案 2B: 改进注意力机制 (推荐)**

在第二层增加多头注意力：

```python
# 当前 conv2 配置
self.conv2 = dglnn.HeteroGraphConv({
    rel: dglnn.GATConv(
        in_feats=hidden_dim * 4,
        out_feats=out_dim,
        num_heads=1,  # ← 问题：只有 1 个头，没有利用注意力优势
        ...
    )
})

# 改为
self.conv2 = dglnn.HeteroGraphConv({
    rel: dglnn.GATConv(
        in_feats=hidden_dim * 4,
        out_feats=out_dim,
        num_heads=4,  # ← 增加到 4 个头
        ...
    )
})
```

**预期效果**: +0.005-0.015
**实施难度**: ⭐ (仅需改一行)

---

**方案 2C: 增加跨关系注意力 (高级)**

在不同关系之间添加注意力机制：

```python
class CrossRelationAttention(nn.Module):
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.num_relations = num_relations

    def forward(self, relation_outputs):
        # relation_outputs: dict of {rel_type: [batch_size, hidden_dim]}
        # 堆叠所有关系的输出
        stacked = torch.stack(list(relation_outputs.values()), dim=1)
        # 应用多头注意力
        attended, _ = self.attention(stacked, stacked, stacked)
        # 返回融合后的表示
        return attended.mean(dim=1)
```

**预期效果**: +0.02-0.03
**实施难度**: ⭐⭐⭐⭐ (需要重构模型)

---

### 方向 3️⃣: 特征工程改进 (优先级: ⭐⭐⭐)

#### 当前特征分析

```
节点特征来源：
1. 图卷积输出: hidden_dim (256)
2. 文本编码: encoded (512)
3. 节点类型: categorical

分类器输入: hidden_dim * 3 = 768
分类器输出: 2 classes
```

**问题**:
- 文本编码信息没有充分利用（只在后期注入）
- 早期和中期没有充分融合多模态信息
- 节点类型信息可能被忽视

#### 改进方案

**方案 3A: 早期多模态融合**

```python
class EarlyFusionGraphModule(nn.Module):
    def __init__(self, graph_dim, text_dim, hidden_dim):
        super().__init__()
        # 融合图特征和文本特征
        self.fusion = nn.Linear(graph_dim + text_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph_features, text_features):
        # graph_features: [num_nodes, graph_dim]
        # text_features: [num_nodes, text_dim]
        combined = torch.cat([graph_features, text_features], dim=1)
        fused = self.fusion(combined)
        fused = self.bn(fused)
        return fused
```

**预期效果**: +0.01-0.02
**实施难度**: ⭐⭐ (需要修改前向传播)

---

**方案 3B: 添加全局图特征**

```python
class GlobalGraphFeature(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, node_features):
        # 计算全局图特征（平均池化）
        global_feat = node_features.mean(dim=0, keepdim=True)
        # 编码全局特征
        encoded_global = self.graph_encoder(global_feat)
        return encoded_global
```

**预期效果**: +0.01-0.02
**实施难度**: ⭐⭐

---

## 三、推荐实施计划

### 优先级排序

| 优先级 | 改进方案 | Macro F1 提升 | 耗时 | 实现难度 |
|-------|--------|-------------|------|--------|
| 1 | 关系类型减少 (1A) | +0.01-0.03 | 10 min | ⭐ |
| 2 | 改进注意力头 (2B) | +0.005-0.015 | 10 min | ⭐ |
| 3 | 增加网络深度 (2A) | +0.02-0.04 | 30 min | ⭐⭐⭐ |
| 4 | 早期多模态融合 (3A) | +0.01-0.02 | 30 min | ⭐⭐ |
| 5 | 跨关系注意力 (2C) | +0.02-0.03 | 60 min | ⭐⭐⭐⭐ |

---

### 建议执行方案

#### **第 1 周 (立即开始)**

**Model 5**: 关系类型优化 (1A)
```bash
# 修改 train.py 第 718 行
rel_names=rel_names_short,  # 改成短列表

# 修改 train.py 第 496 行
model.merge_graphs(g_p1, g_p2, lc, rel_names_short)

# 运行训练 (1.5 小时)
CUDA_VISIBLE_DEVICES=0 python train.py > training_model5_rel_short.log 2>&1
```

**预期**: 0.5611 → 0.57-0.58

---

#### **第 2 周 (如果有改进)**

**Model 6**: 改进注意力头 (2B)
```python
# 修改 build_base_graph_extract.py 中 RGAT 类
# conv2 的配置改为
num_heads=4,  # 从 1 改为 4

# 或者同时增加网络深度 (2A)
# 添加第三层卷积
```

**预期**: 0.57-0.58 → 0.58-0.59

---

#### **第 3 周 (综合优化)**

**Model 7**: 多个改进组合
- 关系类型 (rel_names_short)
- 网络深度增加 (3 层)
- 早期多模态融合

**预期**: → 0.59-0.60+

---

## 四、具体实施指南

### 步骤 1: 关系类型优化 (最简单)

**文件**: `train.py`

```python
# 第 496 行，修改为
model.merge_graphs(g_p1, g_p2, lc, rel_names_short)

# 第 714-718 行，修改为
model = ExplainableHeteroClassifier(
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    rel_names=rel_names_short,  # ← 关键改动
)
```

**运行**:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py > training_model5_short_rel.log 2>&1
```

**验证改动**:
```bash
# 检查是否修改成功
grep "rel_names=rel_names_" train.py
grep "merge_graphs.*rel_names" train.py
```

---

### 步骤 2: 网络深度增加 (中等难度)

**文件**: `build_base_graph_extract.py`

在 RGAT 类的 `forward` 方法中添加第三层：

```python
def forward(self, g, inputs, return_attention=False):
    h_dict = inputs

    # Layer 1
    h = self.conv1(g, h_dict)
    h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

    # Layer 2 (现有)
    h = self.conv2(g, h)
    h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

    # Layer 3 (新增) ← 在这里添加
    # ... 复制 Layer 2 的代码，但输入维度调整

    out = {k: v.squeeze(1) for k, v in h.items()}
    return out
```

---

## 五、预期改进时间表

```
立即 (今天):
  Model 5: 关系类型优化
  预期: 0.5611 → 0.57-0.58
  耗时: 1.5 小时

明天:
  Model 6: 改进注意力 + 网络深度
  预期: 0.57-0.58 → 0.58-0.59
  耗时: 2-3 小时

本周末:
  Model 7: 综合优化
  预期: 0.58-0.59 → 0.59-0.60+
  耗时: 3-4 小时

总耗时: 6-8 小时
```

---

## 六、风险与注意事项

### 关系类型减少的风险

**风险**: 可能丢失重要的语义关系

**缓解方案**:
1. 先测试 rel_names_short，看效果
2. 如果性能下降，尝试自定义的 15-18 个关系
3. 逐步实验不同的关系组合

### 网络深度增加的风险

**风险**: 过拟合、梯度消失、训练不稳定

**缓解方案**:
1. 增加 dropout（当前 0.1，改为 0.15-0.2）
2. 使用残差连接（已有）
3. 添加 BatchNorm
4. 使用早停策略

### 多模态融合的风险

**风险**: 维度爆炸、计算复杂度增加

**缓解方案**:
1. 使用瓶颈层压缩特征
2. 逐层融合而不是全量融合

---

## 七、关键代码变化清单

### Model 5: 关系优化
- [ ] train.py 第 496 行: `rel_names_long` → `rel_names_short`
- [ ] train.py 第 718 行: `rel_names=rel_names_long` → `rel_names=rel_names_short`

### Model 6: 网络改进
- [ ] build_base_graph_extract.py, RGAT 类: conv2 的 num_heads: 1 → 4
- [ ] 或添加 conv3 层（完整网络深度增加）

### Model 7: 综合优化
- [ ] 结合 Model 5 + Model 6 的所有改动
- [ ] 可选：添加多模态融合

---

## 八、性能对比预期

| 模型 | 关系类型 | 网络深度 | Macro F1 | vs Baseline |
|------|--------|--------|----------|-----------|
| Model 1 | 20 types | 2 layers | 0.5494 | baseline |
| Model 3 | 20 types | 2 layers | 0.5611 | +0.0117 |
| Model 5 | 9 types | 2 layers | 0.57-0.58 | +0.01-0.03 |
| Model 6 | 9 types | 3 layers | 0.58-0.59 | +0.03-0.04 |
| Model 7 | 9 types | 3 layers + fusion | 0.59-0.60+ | +0.04-0.05+ |

---

## 推荐立即执行

**今天**:
1. 备份 train.py: `cp train.py train.py.backup_before_model5`
2. 修改 train.py 两行（关系类型）
3. 运行 Model 5
4. 对比结果

如果 Model 5 有改进 > 0.005，立即进行 Model 6。

---

**现在就开始吧！这次改动是根本性的，预期会有明显效果。** 🚀
