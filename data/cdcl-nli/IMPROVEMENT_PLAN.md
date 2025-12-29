# Macro F1 改进方案报告

**生成日期**: 2025-12-27

## 一、现状分析

### 历史实验结果对比

| 指标 | Model 1 | Model 2 | 对比 |
|------|---------|---------|------|
| 初始 Macro F1 | 0.4361 | 0.1848 | Model 1 更好 |
| 最终 Macro F1 | **0.5494** | 0.5284 | Model 1 更优 |
| 最优轮次 | Epoch 18 | Epoch 23 | Model 1 收敛快 |
| 最终 Accuracy | 0.7411 | 0.5811 | Model 1 更好 |
| 类别分布问题 | 前期严重不平衡 | 初期反向预测 | 两者都有 |

### 关键问题识别

#### 1. **类别不平衡严重** (数据集问题)
- 标签分布: Class 0: 696 个 (77.3%), Class 1: 204 个 (22.7%)
- 比例: 3.4:1 (高度不平衡)
- 对 Macro F1 的影响: 由于是宏平均，少数类的性能权重更大

#### 2. **模型早期学习困难** (Model 1)
- **Epoch 0-11**: 模型只预测 Class 0，无法捕捉 Class 1
- **Epoch 12**: 才开始有 Class 1 的预测 (13 个)
- **Epoch 15**: Class 1 预测才达到 97 个
- **问题根因**: 初始化可能过于保守，或学习率过低

#### 3. **早期预测反向崩溃** (Model 2)
- **Epoch 0-5**: 模型全部预测 Class 1，导致 Macro F1 极低 (0.1848)
- **Epoch 7**: 才逐步开始学习正常的类别分布
- **问题根因**: 类别权重设置不当，或梯度爆炸

#### 4. **Recall & Precision 权衡**
- 两个模型都存在 Recall 和 Precision 的不平衡
- Model 1 最优状态: Precision 0.5798, Recall 0.5485 (略高于 Recall)
- Model 2 最优状态: Precision 0.5484, Recall 0.5680 (略低于 Recall)

---

## 二、改进方案

### **方案 A: 优化数据和类别权衡** (优先级: ⭐⭐⭐⭐⭐)

#### A1. 调整 Focal Loss 的类别权重
**当前配置**: `alpha=[0.5, 2.0]` (Class 0: 0.5, Class 1: 2.0)

**改进方案**:
```python
# 计算动态权重
num_class_0 = 696
num_class_1 = 204
total = num_class_0 + num_class_1

# 方案 1: 反向权重比例
weight_0 = num_class_1 / total  # 0.226
weight_1 = num_class_0 / total  # 0.774
# 结果: alpha=[0.226, 0.774] 或 alpha=[1.0, 3.4] (标准化)

# 方案 2: 平方根加权 (缓和过度补偿)
weight_0 = (num_class_1 / total) ** 0.5  # 0.475
weight_1 = (num_class_0 / total) ** 0.5  # 0.880
# 结果: alpha=[0.475, 0.880] 或标准化后: alpha=[1.08, 2.0]

# 推荐使用: alpha=[1.0, 3.4] 或 alpha=[1.2, 3.0]
```

**实施步骤**:
1. 修改 `train.py` 第 128 行的 Focal Loss 初始化
2. 建议尝试多个权重组合: `[1.0, 3.4]`, `[1.2, 3.0]`, `[1.5, 2.5]`
3. 记录每个配置的结果

#### A2. 调整 Gamma 参数
**当前配置**: `gamma=2.0`

**改进方案**:
- 尝试 `gamma=2.5` 或 `gamma=3.0` (更强的难样本关注)
- 对于不平衡数据，较高的 gamma 值更有效

#### A3. 增加/调整 Label Smoothing
**当前配置**: `label_smoothing=0.1`

**改进方案**:
- 尝试 `label_smoothing=0.05` (降低过度平滑)
- 或保持 0.1，但结合其他改进

---

### **方案 B: 优化初始化和学习率** (优先级: ⭐⭐⭐⭐)

#### B1. 改进权重初始化
**问题**: 模型前期无法预测少数类

**解决方案**:
```python
# 在 train.py 中添加初始化代码
def init_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, std=0.02)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

#### B2. 调整学习率与预热
**当前配置**: 需要检查 `train.py` 中的学习率设置

**改进方案**:
- 增加预热步数: 从 3360 改为 5000-6000
- 降低初始学习率: 从默认值改为更小的值
- 使用 `get_cosine_schedule_with_warmup` (已在用，保持)

#### B3. 实施分阶段学习率策略
```python
# 伪代码
if epoch < 5:
    # 早期: 保守学习，关注主要类别特征
    alpha_weights = [0.3, 2.0]
elif epoch < 10:
    # 中期: 逐步提高对少数类的关注
    alpha_weights = [0.5, 2.5]
else:
    # 后期: 全力关注两个类别的平衡
    alpha_weights = [1.0, 3.0]
```

---

### **方案 C: 数据增强和采样策略** (优先级: ⭐⭐⭐⭐)

#### C1. 过采样少数类 (Oversampling)
**实施步骤**:
```python
# 在 data_loader.py 中实施
from imblearn.over_sampling import RandomOverSampler

# 训练集中过采样 Class 1
oversampler = RandomOverSampler(
    sampling_strategy=0.8,  # 使少数类达到多数类的 80%
    random_state=42
)
```

#### C2. 欠采样多数类 (Undersampling)
**组合使用过采样和欠采样**:
```python
# 保留多数类的 90%，过采样少数类到 70%
sampling_strategy = 0.777  # 204 / (696 * 0.9) ≈ 0.33
```

#### C3. 混合采样策略 (推荐)
```python
# SMOTE + RandomUndersampler 的组合
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.8))
])
```

---

### **方案 D: 模型架构改进** (优先级: ⭐⭐⭐)

#### D1. 添加注意力机制权重
```python
# 在 build_base_graph_extract.py 中的分类头部分
class BalancedClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        # 添加类别特定的注意力权重
        self.class_weights = nn.Parameter(
            torch.ones(num_classes) / num_classes
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        # 应用可学习的类别权重
        logits = logits * self.class_weights.unsqueeze(0)
        return logits
```

#### D2. 使用ArcFace或CosFace损失
**替代 Focal Loss** (对于二分类任务):
```python
# 添加更强的类别间隔约束
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(in_features, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # 计算 cosine 相似度
        cos = F.linear(F.normalize(features), F.normalize(self.weight))
        # 应用 margin 和 scale
        ...
```

---

### **方案 E: 评估指标和监控** (优先级: ⭐⭐⭐⭐)

#### E1. 不只关注 Macro F1，同时监控
```python
# 在评估时记录
metrics = {
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    'f1_class_0': f1_score(y_true, y_pred, labels=[0], average='macro'),
    'f1_class_1': f1_score(y_true, y_pred, labels=[1], average='macro'),
    'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
}
```

#### E2. 使用 Matthews 相关系数 (MCC)
```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
# MCC 更适合不平衡分类问题，范围 [-1, 1]
```

#### E3. 早停策略改进
```python
# 不仅基于 macro F1，同时检查：
# - f1_class_1 (少数类性能)
# - 防止过拟合的验证损失
# 建议: 基于加权指标 0.6*macro_f1 + 0.4*f1_class_1
```

---

## 三、推荐的实验计划

### **实验 1: 权重优化** (用时: 2-3 小时)
```yaml
配置:
  - focal_loss_alpha: [[1.0, 3.4], [1.2, 3.0], [1.5, 2.5], [2.0, 2.0]]
  - gamma: [2.0, 2.5, 3.0]
  - label_smoothing: [0.05, 0.1, 0.15]

期望结果: Macro F1 > 0.56
```

### **实验 2: 学习率调度** (用时: 1-2 小时)
```yaml
配置:
  - warmup_steps: [3360, 5000, 6000]
  - base_lr: [1e-5, 2e-5, 5e-5]
  - lr_decay: cosine (保持)

期望结果: 更快收敛，Macro F1 > 0.55
```

### **实验 3: 数据平衡** (用时: 3-4 小时)
```yaml
方案:
  - 过采样 Class 1 到 80%
  - 或使用 SMOTE + UnderSampler
  - 或动态采样权重

期望结果: Macro F1 > 0.57，更稳定的收敛
```

### **实验 4: 综合优化** (用时: 4-6 小时)
```yaml
组合:
  - Focal Loss: alpha=[1.2, 3.0], gamma=2.5
  - 学习率: warmup=5000, base_lr=2e-5
  - 数据平衡: SMOTE 过采样 0.5
  - 初始化: xavier_uniform

期望结果: Macro F1 > 0.58-0.60
```

---

## 四、实施建议

### 优先顺序:
1. **第一步**: 调整 Focal Loss 权重 (最快，影响最大)
2. **第二步**: 增加学习率预热步数
3. **第三步**: 实施数据采样策略
4. **第四步**: 调整初始化和注意力机制

### 记录方式:
```python
# 在 train.py 中添加配置日志
experiment_config = {
    'focal_loss_alpha': [1.0, 3.4],
    'focal_loss_gamma': 2.5,
    'label_smoothing': 0.1,
    'warmup_steps': 5000,
    'oversampling_strategy': 0.8,
    'date': '2025-12-27',
    'notes': '调整权重并增加数据平衡'
}

logging.info(f"Experiment config: {json.dumps(experiment_config, indent=2)}")
```

### 保存结果:
```
training_model3.log  <- Experiment 1: 权重优化
training_model4.log  <- Experiment 2: 学习率调整
training_model5.log  <- Experiment 3: 数据平衡
training_model6.log  <- Experiment 4: 综合优化
```

---

## 五、关键代码位置

| 需要修改的位置 | 文件 | 行号 | 改进内容 |
|---------------|------|------|---------|
| Focal Loss 初始化 | train.py | 128 | 调整 alpha, gamma |
| 数据加载器 | data_loader.py | - | 添加过采样逻辑 |
| 权重初始化 | train.py | - | 改进模型权重初始化 |
| 学习率调度 | train.py | - | 增加预热步数 |
| 评估指标 | cal_scores.py | - | 添加更多监控指标 |
| 分类头部分 | build_base_graph_extract.py | - | 添加注意力机制 |

---

## 六、预期改进空间

- **保守估计**: Macro F1: 0.5494 → 0.56-0.57 (+0.01-0.02)
- **中等目标**: Macro F1: 0.5494 → 0.57-0.59 (+0.02-0.04)
- **激进目标**: Macro F1: 0.5494 → 0.60+ (+0.05+)

---

## 七、风险与注意事项

1. **过度拟合风险**: 增加少数类权重可能导致过拟合，需要监控验证集性能
2. **梯度不稳定**: 权重比例过大（如 1:5）可能导致梯度波动，需要梯度裁剪
3. **数据泄露**: 如使用 SMOTE，需确保仅在训练集上应用，不影响验证/测试集
4. **超参数敏感**: 不同的数据和模型对权重设置敏感，需要系统实验

---

**建议立即实施**: 修改 alpha 权重，这是最直接有效的改进
