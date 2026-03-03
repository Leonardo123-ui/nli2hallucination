# LLM+CDCL 图节点增强实现指南

## 整体方案

采用**早期融合（Early Fusion）**策略，在图神经网络推理之前就将LLM的分析结果融入图节点特征。

### 核心思想

```
原始文本对 → LLM分析 → 提取关键词 → 匹配图节点 → 计算权重 → 增强特征 → GNN推理 → 最终预测
          rationale                                        加权乘法
```

## 实现文件结构

### 1. `llm_graph_augmentation.py` - 图增强模块（已创建）

**核心函数：**

```python
def build_augmented_graph(
    g_premise: dgl.DGLGraph,
    g_hypothesis: dgl.DGLGraph,
    rationale: str,
    method: str = "multiply"
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """
    使用LLM的rationale增强前提和假设的图节点特征

    流程：
    1. 从rationale提取关键词（top-10）
    2. 将关键词匹配到图节点中的文本
    3. 为匹配的节点计算权重（1.0 for关键, 0.5 for普通）
    4. 将权重融入特征向量
    """
```

**权重融合方法：**

- **multiply** (默认，推荐): `features *= weight`
  - 直观明确，关键节点特征被放大1.0倍，普通节点0.5倍

- **concat**: 权重作为额外维度拼接到特征后
  - 让模型自己学习如何使用权重

- **gating**: 门控融合，强调关键节点
  - `features *= (0.8 + 0.4 * weight)`

### 2. `train.py` - 训练脚本修改

**修改点1：导入和初始化 (行 43-52)**
```python
# LLM+CDCL相关导入
from main import QwenLLMClient, QwenConfig, ModelDeployType
```

**修改点2：main()函数中的LLM客户端初始化 (行 719-734)**
```python
# 初始化Qwen LLM客户端
llm_client = QwenLLMClient(config=QwenConfig(
    model_name="/mnt/second/yuanmengying/qwen3-8b",
    deploy_type=ModelDeployType.HUGGINGFACE,
))
```

**修改点3：process_batch()函数中的增强逻辑 (行 524-555)**
```python
# 应用LLM增强（仅在训练阶段）
if llm_client is not None and stage == "train":
    from llm_graph_augmentation import augment_batch_graphs

    # 为每个样本生成rationale
    rationales = [llm_client.call_llm(...) for i in range(len(graph1))]

    # 应用增强
    graph1, graph2 = augment_batch_graphs(graph1, graph2, rationales, method="multiply")
```

**修改点4：train_epoch()函数签名 (行 638)**
```python
def train_epoch(..., llm_client=None):
    # 在调用process_batch时传递llm_client
    batch_metrics = process_batch(..., llm_client=llm_client)
```

## 关键词提取策略

从rationale中提取关键词的步骤：

1. **文本清洁**：移除特殊字符，保留中英文和空格
2. **分词**：
   - 英文按空格分割，提取单词
   - 中文提取长度为2-4的词组（简化方案）
3. **过滤**：移除停用词（英文/中文）
4. **排序**：按词频选择top-10关键词

**示例：**
```
Rationale: "Summary中提到的'机器学习'在Source中没有直接提及，这是不一致之处。
           另外Summary强调了'系统能力'而Source重点讨论的是'架构设计'。"

提取的关键词: ['机器学习', '不一致', '系统', '能力', '架构', '设计', ...]
```

## 节点匹配和权重计算

```python
def match_keywords_to_nodes(graph, keywords):
    """
    为每个图节点计算权重

    base_weight = 0.5  # 普通节点
    keyword_weight = 1.0  # 包含关键词的节点

    匹配分数 = 节点中出现的关键词数量 / 总关键词数量
    最终权重 = base_weight + 匹配分数 * (keyword_weight - base_weight)

    示例：
    - 节点包含3个关键词（总共10个）→ 权重 = 0.5 + 0.3 * 0.5 = 0.65
    - 节点包含1个关键词            → 权重 = 0.5 + 0.1 * 0.5 = 0.55
    - 节点不包含关键词             → 权重 = 0.5
    """
```

## 特征融合方式对比

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **multiply** | `f' = f * w` | 简单直观，数据无膨胀 | 固定的权重影响 |
| **concat** | `f' = [f, w]` | 模型可自适应学习 | 特征维度增加 |
| **gating** | `f' = f * (α + βw)` | 可控的强度调节 | 需要调参 |

## 性能预期

基于"Rationale Injection"原理：

1. **信息流优化**：LLM的关键分析能从GNN的第一层就指导信息流
2. **注意力增强**：RGAT的attention机制会自适应学习权重的重要性
3. **预期提升**：F1 score 提升 1-3% （取决于rationale质量）

## 故障排除

### 1. 导入错误
```
ImportError: cannot import name 'QwenLLMClient'
```
**解决**：确保main.py和llm_graph_augmentation.py在同一目录

### 2. 节点文本解码失败
```
Warning: 解码文本失败
```
**原因**：text_encoded可能未正确存储
**解决**：检查图构建时ndata["text_encoded"]是否正确设置

### 3. LLM调用超时
```
LLM增强失败
```
**解决**：增加timeout或跳过LLM增强（训练仍会继续）

## 使用建议

### 开发调试
```bash
# 1. 用小数据集快速验证
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 2 --batch-size 4

# 2. 查看日志中的增强信息
grep "LLM增强" logs/training_model11_llm.log
```

### 生产训练
```bash
# 完整训练（50 epoch）
CUDA_VISIBLE_DEVICES=0 python train.py > logs/training_model11_final.log 2>&1 &
```

### 性能对比
```bash
# 对比有/无LLM增强的效果
# Model 11 (with LLM):     logs/training_model11_llm.log
# Model 10 (without LLM):  logs/training_model10.log

# 查看最终F1
echo "=== Model 11 (LLM增强) ==="
grep "f1_macro_cli:" logs/training_model11_llm.log | tail -1

echo "=== Model 10 (无增强) ==="
grep "f1_macro_cli:" logs/training_model10.log | tail -1
```

## 下一步优化方向

1. **关键词提取优化**
   - 集成jieba分词库以获得更精确的中文分词
   - 使用TF-IDF替代简单词频
   - 集成词性标注，优先选择名词/动词

2. **权重计算优化**
   - 基于编辑距离的模糊匹配
   - 字义相似度匹配（使用词向量）
   - 动态权重范围（而非固定的0.5-1.0）

3. **融合方式优化**
   - 图注意力权重直接融合：`attention = attention * weight`
   - 消息传递阶段的融合
   - 多头融合（不同权重应用到不同head）

4. **Rationale生成优化**
   - 使用专门的NLI分析prompt设计
   - 集成外部知识库
   - 多轮LLM对话获得深层分析
