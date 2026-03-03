# LLM+CDCL 图节点增强实现完成报告

## 实现概述

已完成**早期融合（Early Fusion）**策略的 LLM+CDCL 图节点增强系统实现。将LLM的分析结果（Rationale）在GNN推理之前融入图节点特征，用于NLI（自然语言推断）判别任务。

---

## 核心设计方案

### 整体流程

```
输入 (Source, Hypothesis)
          ↓
    LLM 生成 Rationale
    ("识别出关键不一致处")
          ↓
    提取关键词 + 分词
    (Top-10 关键词)
          ↓
    匹配图节点文本
    (计算每个节点的权重)
          ↓
    特征加权融合
    (features *= weight)
          ↓
   增强后的图 → GNN推理
    (RGAT多层传播)
          ↓
  最终预测 (Entailment/Contradiction)
```

### 关键特性

| 特性 | 实现方式 |
|------|---------|
| **LLM集成** | QwenLLMClient，支持HuggingFace本地加载 |
| **关键词提取** | 正则+词频，中英文分词，停用词过滤 |
| **节点匹配** | 文本相似度匹配（子串包含关系） |
| **权重融合** | multiply (默认) / concat / gating |
| **应用阶段** | 训练阶段（仅在 stage="train"） |
| **失败容错** | 增强失败时自动回退原始图 |

---

## 文件结构

```
/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/
├── llm_graph_augmentation.py    # ✨ 新增：图增强核心模块
├── train.py                      # ✓ 修改：集成LLM增强
├── main.py                       # ✓ 已有：QwenLLMClient
├── detector.py                   # ✓ 已有：AugmentedDiscriminator
├── requirements.txt              # ✓ 已有：依赖列表
├── LLM_AUGMENTATION_GUIDE.md    # ✨ 新增：详细指南
└── logs/
    ├── training_model11_llm.log  # 正在运行的训练日志
    └── ...
```

---

## 关键实现细节

### 1. `llm_graph_augmentation.py` - 图增强模块

**核心函数签名：**

```python
def build_augmented_graph(
    g_premise: dgl.DGLGraph,
    g_hypothesis: dgl.DGLGraph,
    rationale: str,
    method: str = "multiply"
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
```

**内部处理流程：**

1. **关键词提取** (`extract_keywords_from_rationale`)
   - 输入：LLM的分析文本 (rationale)
   - 输出：Top-10 关键词列表
   - 处理：正则清洁 → 分词 → 停用词过滤 → 词频排序

2. **节点文本解码** (`decode_text_from_encoded`)
   - 从 ndata["text_encoded"] 中解码Base64文本
   - 支持节点级别的文本访问

3. **关键词-节点匹配** (`match_keywords_to_nodes`)
   - 对每个节点计算匹配分数
   - 权重公式：`weight = 0.5 + match_score * 0.5`
   - 匹配分数 = (节点中出现的关键词数) / (总关键词数)

4. **特征融合** (`apply_weights_to_features`)
   - **multiply (推荐)**：`features *= weight`
   - **concat**：`features = [features, weight]`
   - **gating**：`features *= (0.8 + 0.4 * weight)`

### 2. `train.py` 集成修改

**修改位置 1：导入 (行 43-52)**
```python
from main import QwenLLMClient, QwenConfig, ModelDeployType
```

**修改位置 2：LLM初始化 (行 719-734)**
```python
llm_client = QwenLLMClient(config=QwenConfig(
    model_name="/mnt/second/yuanmengying/qwen3-8b",
    deploy_type=ModelDeployType.HUGGINGFACE,
))
```

**修改位置 3：process_batch 增强逻辑 (行 512-555)**
```python
def process_batch(..., llm_client=None):
    # ... 现有的图增强逻辑 ...

    # 应用LLM增强（仅训练阶段）
    if llm_client is not None and stage == "train":
        from llm_graph_augmentation import augment_batch_graphs

        # 为每个样本生成rationale
        rationales = [llm_client.call_llm(...) for i in range(len(graph1))]

        # 应用增强到所有图
        graph1, graph2 = augment_batch_graphs(graph1, graph2, rationales, method="multiply")
```

**修改位置 4：train_epoch 签名 (行 638)**
```python
def train_epoch(..., llm_client=None):
    # 调用时传递llm_client
    batch_metrics = process_batch(..., llm_client=llm_client)
```

**修改位置 5：train_epoch 调用 (行 843-854)**
```python
train_losses = train_epoch(
    ...,
    llm_client=llm_client,
)
```

---

## 权重融合方式对比

### 方式对比表

| 方法 | 公式 | 特点 | 推荐场景 |
|------|------|------|---------|
| **multiply** | `f' = f * w` | 直观、无膨胀、固定权重 | ✓ 通用（默认） |
| **concat** | `f' = [f, w]` | 维度增加、自适应学习 | 特征充足时 |
| **gating** | `f' = f * (α + βw)` | 可控强度、需调参 | 微调阶段 |

### 权重分布示例

给定 rationale: "Summary中缺少对'机器学习'的讨论，这是关键不一致"

```
关键词: ['机器学习', '讨论', '关键', '不一致', ...]

节点1 "机器学习是AI的重要分支"
  └─ 匹配关键词：['机器学习'] → 权重 = 0.6

节点2 "模型使用随机梯度下降"
  └─ 匹配关键词：[] → 权重 = 0.5

节点3 "这导致了关键性能指标提升"
  └─ 匹配关键词：['关键'] → 权重 = 0.55
```

---

## 性能指标预期

基于"理由注入"策略的原理：

| 指标 | 无增强 | 有增强 | 提升 |
|------|--------|--------|------|
| F1 Score | ~0.56 | ~0.57-0.59 | **+1-3%** |
| Precision | ~0.55 | ~0.56-0.58 | **+1-3%** |
| Recall | ~0.57 | ~0.58-0.60 | **+1-3%** |

*预期提升取决于 LLM rationale 的质量和图结构的复杂度*

---

## 运行方式

### 开发调试

```bash
# 1. 快速验证（小数据集，2 epochs）
CUDA_VISIBLE_DEVICES=0 python train.py &

# 2. 查看增强日志
grep "LLM增强\|✓" logs/training_model11_llm.log
```

### 生产训练

```bash
# 完整训练（50 epochs，后台运行）
CUDA_VISIBLE_DEVICES=0 python train.py > logs/training_model11_final.log 2>&1 &

# 监控进度
tail -f logs/training_model11_final.log | grep "Epoch\|f1_macro"
```

### 性能对比

```bash
# 查看增强前后的F1分数
echo "=== Model 11 (with LLM Enhancement) ==="
grep "f1_macro_cli:" logs/training_model11_llm.log | tail -5

echo "=== Model 10 (baseline) ==="
grep "f1_macro_cli:" logs/training_model10.log | tail -5
```

---

## 故障排除

### 问题 1：ImportError

```
ImportError: cannot import name 'QwenLLMClient'
```

**原因**：main.py 不在同一目录
**解决**：确保 llm_graph_augmentation.py 和 train.py 在同一目录，且 main.py 可访问

### 问题 2：节点文本解码失败

```
Warning: 解码文本失败
```

**原因**：ndata["text_encoded"] 可能损坏或格式不对
**解决**：检查图构建代码，确保 text_encoded 正确设置

### 问题 3：LLM 超时

```
LLM增强失败: ...timeout...
```

**原因**：Qwen 模型加载/推理耗时长
**解决**：增加 timeout 或跳过某些样本的增强（训练继续）

### 问题 4：内存溢出

```
CUDA out of memory
```

**原因**：并行增强和推理消耗大量内存
**解决**：减少 batch_size 或跳过 LLM 增强（设 llm_client=None）

---

## 优化建议

### 短期优化（立即可做）

1. **关键词提取改进**
   - 集成 jieba 分词库获得精确分词
   - 使用 TF-IDF 替代简单词频
   - 词性标注优先选择名词/动词

2. **权重计算改进**
   - 基于编辑距离的模糊匹配
   - 同义词识别（使用词向量）
   - 动态权重范围 (vs 固定0.5-1.0)

### 中期优化（需要调研）

3. **融合方式改进**
   - 图注意力直接融合：`attention *= weight`
   - 消息传递阶段的融合
   - 多头融合（不同weight应用到不同head）

4. **Rationale 生成优化**
   - 专门的 NLI 分析 prompt
   - 多轮 LLM 对话获得深层分析
   - 集成外部知识库

### 长期优化（架构改进）

5. **端到端学习**
   - 训练 LLM prompt 参数
   - 学习权重计算函数
   - 联合优化 LLM + GNN

---

## 实验验证计划

### Phase 1：基线验证 ✓ (进行中)
- Model 11 (with enhancement): 50 epochs
- 对比 Model 10 (baseline)

### Phase 2：消融研究 (待做)
```
- 不同关键词提取方法
- 不同权重融合方式
- 不同 rationale prompt
```

### Phase 3：超参调优 (待做)
```
- 权重范围：[0.3-1.0] vs [0.5-1.0]
- 关键词数：top-5 vs top-10 vs top-20
- 融合权重：α, β 参数调优
```

---

## 总结

✅ **已完成实现：**
- LLM 集成（QwenLLMClient，HuggingFace 支持）
- 图增强模块（关键词提取、节点匹配、特征融合）
- 训练流程集成（process_batch、train_epoch 修改）
- 详细文档和指南

⏳ **进行中：**
- Model 11 完整训练（50 epochs）
- 性能对比验证

📋 **后续计划：**
- 消融研究（各个模块的贡献度）
- 超参调优
- 长期优化（端到端学习）

