# 改进模型使用指南

## 📁 文件说明

### 核心文件

1. **IMPROVEMENT_PLAN.md** - 详细的改进计划文档
2. **improved_losses.py** - 改进的损失函数模块
3. **improved_model.py** - 改进的模型架构模块
4. **train_improved.py** - 改进的训练脚本
5. **run_improved.sh** - 快速启动脚本

### 主要改进点

#### Phase 1: 快速改进
- ✅ 改进的Focal Loss (alpha=[0.5, 4.5], gamma=3.0)
- ✅ 对比学习损失
- ✅ 动态阈值调优
- ✅ LLM晚期融合（架构已支持）

#### Phase 2: 图结构优化
- ✅ 层次注意力聚合
- ✅ 关系重要性学习
- ✅ 跨图交互注意力
- ✅ 多尺度图池化

## 🚀 快速开始

### 1. 环境要求

```bash
python >= 3.8
torch >= 1.12.0
dgl >= 0.9.0
scikit-learn
tqdm
numpy
```

### 2. 运行训练

#### 方法1: 使用快速启动脚本

```bash
chmod +x run_improved.sh
./run_improved.sh
```

#### 方法2: 直接运行

```bash
CUDA_VISIBLE_DEVICES=0 python train_improved.py
```

### 3. 训练配置

默认配置（可在 `train_improved.py` 中修改）：

```python
# 损失函数
Focal Loss: alpha=[0.5, 4.5], gamma=3.0, label_smoothing=0.05
Contrastive Loss: temperature=0.5, margin=0.5
Loss weights: focal=1.0, contrast=0.1

# 训练参数
epochs = 20
batch_size = 7
learning_rate = 0.001
warmup_ratio = 0.1
patience = 5 (早停)

# 优化器
optimizer = AdamW
weight_decay = 1e-4
scheduler = CosineAnnealingWarmRestarts
```

## 📊 预期性能提升

| 阶段 | Macro F1 | 相对提升 |
|------|----------|----------|
| Baseline | 53.41% | - |
| Phase 1+2 | 65-69% | +22-29% |

## 🔍 模型架构说明

### 改进的异构图分类器 (ImprovedHeteroClassifier)

```
输入: premise图 + hypothesis图 + 词汇链
  ↓
第一层RGAT (关系重要性学习)
  ↓
跨图交互注意力
  ↓
第二层RGAT
  ↓
多尺度图池化 (avg + max + attn + set2set)
  ↓
图表示拼接 [z_premise, z_hypothesis, z_premise * z_hypothesis]
  ↓
混合分类器 (支持LLM特征融合)
  ↓
输出: 分类logits + 图表示 (用于对比学习)
```

### 损失函数组合

```python
Total Loss = focal_weight * Focal Loss + contrast_weight * Contrastive Loss
           = 1.0 * Focal Loss + 0.1 * Contrastive Loss
```

## 📈 训练监控

### 日志文件

训练过程会生成以下文件：

```
checkpoints/improved_model_YYYYMMDD_HHMMSS/
├── best_model.pt              # 最优模型检查点
└── config.yaml                # 训练配置

training_improved_YYYYMMDD_HHMMSS.log  # 训练日志
```

### 关键指标监控

训练日志中会输出：

1. **每个epoch的损失**
   - cls_loss (分类损失)
   - contrast_loss (对比学习损失)
   - total_loss (总损失)

2. **每个epoch的评估指标**
   - accuracy
   - f1_macro (主要优化目标)
   - f1_micro
   - precision
   - recall

3. **预测分布诊断**
   - Prediction distribution (模型预测的类别分布)
   - Label distribution (真实标签分布)

4. **阈值调优结果**
   - 最优阈值
   - 阈值调优后的F1分数

## 🔧 进阶使用

### 1. 调整损失权重

在 `train_improved.py` 中修改：

```python
focal_weight = 1.0      # Focal Loss权重
contrast_weight = 0.1   # 对比学习权重 (建议范围: 0.05-0.2)
```

### 2. 修改Focal Loss参数

在 `train_improved.py` 中修改：

```python
focal_loss = ImprovedFocalLoss(
    alpha=[0.5, 4.5],      # 类权重 (可尝试 [0.4, 4.8])
    gamma=3.0,             # Focal参数 (可尝试 2.5-3.5)
    label_smoothing=0.05   # 标签平滑 (可尝试 0-0.1)
)
```

### 3. 调整模型深度

在 `improved_model.py` 的 `ImprovedHeteroClassifier` 中：

```python
# 添加更多RGAT层
self.rgat3 = RelationAwareRGAT(...)

# 在forward中添加
h_pre = self.rgat3(g_pre, h_pre)
h_hyp = self.rgat3(g_hyp, h_hyp)
```

### 4. 修改跨图注意力头数

```python
self.cross_attn = CrossGraphAttention(
    hidden_dim=1024,
    num_heads=8  # 可尝试 4, 8, 16
)
```

## 🐛 调试技巧

### 1. 检查梯度

```python
# 在训练过程中添加
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 2. 可视化注意力权重

```python
# 在模型forward后
g_pre, g_hyp = self.cross_attn(g_pre, g_hyp)
# cross_attn 返回注意力权重，可以保存并可视化
```

### 3. 监控关系重要性

```python
# 训练完成后
rel_importance = model.rgat1.get_relation_importance()
print("Top-5 Important Relations:")
for rel, weight in rel_importance[:5]:
    print(f"  {rel}: {weight:.4f}")
```

## 📝 实验记录

建议使用以下表格记录实验：

| 实验ID | 配置变化 | Train F1 | Val F1 | Test F1 | 备注 |
|--------|---------|----------|--------|---------|------|
| exp_improved_001 | baseline | - | - | - | 初始版本 |
| exp_improved_002 | alpha=[0.4,4.8] | - | - | - | 更激进的权重 |
| exp_improved_003 | contrast_weight=0.2 | - | - | - | 增加对比损失 |

## ⚠️ 常见问题

### 1. CUDA Out of Memory

**解决方案：**
- 减小batch size (7 → 5 → 3)
- 减小hidden_dim (1024 → 512)
- 使用梯度累积

### 2. 训练不稳定 / Loss为NaN

**解决方案：**
- 检查学习率是否过大
- 增加梯度裁剪 (max_norm=0.5)
- 检查输入数据是否有异常值

### 3. F1提升不明显

**可能原因：**
- 对比学习权重过小/过大，尝试调整
- 模型过拟合，增加dropout
- 数据增强不足，考虑SMOTE

## 📚 参考文献

1. Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
2. Contrastive Learning: Chen et al. "A Simple Framework for Contrastive Learning" ICML 2020
3. Graph Attention: Veličković et al. "Graph Attention Networks" ICLR 2018

## 🤝 贡献

如有问题或建议，请联系：Yuan Mengying

---

**最后更新时间：** 2026-02-12
