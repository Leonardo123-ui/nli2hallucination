# 实验配置模板

## 实验 1: Focal Loss 权重优化
**目标**: 通过调整类别权重来改善少数类学习

### 配置变量

```python
# config_exp1.yaml
experiment_name: "exp1_focal_loss_weights"
date: "2025-12-27"

# Focal Loss 配置
focal_loss:
  # 当前使用: [0.5, 2.0]
  # 尝试配置:
  config_1:
    alpha: [1.0, 3.4]  # 基于类别比例倒数
    gamma: 2.0
    label_smoothing: 0.1

  config_2:
    alpha: [1.2, 3.0]  # 稍微缓和的比例
    gamma: 2.5
    label_smoothing: 0.1

  config_3:
    alpha: [1.5, 2.5]  # 更温和的差异
    gamma: 2.0
    label_smoothing: 0.15

# 其他保持不变
seed: 42
epochs: 25
batch_size: 32
```

### 执行步骤

1. 修改 `train.py` 第 128 行：
```python
# 当前
criterion = FocalLoss(alpha=[0.5, 2.0], gamma=2.0, label_smoothing=0.1)

# 改为 (尝试 config_1)
criterion = FocalLoss(alpha=[1.0, 3.4], gamma=2.0, label_smoothing=0.1)
```

2. 运行训练：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py > training_model3_exp1_config1.log 2>&1
```

3. 记录结果：在下表中填写最终 Macro F1

### 结果对比表

| 配置 | Alpha | Gamma | Label Smooth | 最终 Macro F1 | 备注 |
|------|-------|-------|--------------|--------------|------|
| 当前 | [0.5, 2.0] | 2.0 | 0.1 | **0.5494** | 基准 |
| Config 1 | [1.0, 3.4] | 2.0 | 0.1 | **待测试** | 基于比例倒数 |
| Config 2 | [1.2, 3.0] | 2.5 | 0.1 | **待测试** | 增加 gamma |
| Config 3 | [1.5, 2.5] | 2.0 | 0.15 | **待测试** | 增加平滑 |

---

## 实验 2: 学习率和预热优化
**目标**: 通过更好的学习率调度帮助模型早期捕捉少数类

### 需要修改的配置

在 `train.py` 中找到学习率相关代码（通常在使用 `get_cosine_schedule_with_warmup` 的地方）：

```python
# 当前配置（需要从代码中确认）
total_steps = 33600
warmup_steps = 3360  # 总步数的 10%

# 改进建议
# 方案 1: 增加预热步数
warmup_steps = 5000  # 15% 的总步数

# 方案 2: 更激进的预热
warmup_steps = 6000  # 18% 的总步数
```

### 修改步骤

1. 在 `train.py` 中找到 `get_cosine_schedule_with_warmup` 的调用
2. 将 `warmup_steps=3360` 改为其他值
3. 运行对比实验

### 实验网格

| 实验 | 预热步数 | 预期效果 |
|------|--------|---------|
| 当前 | 3360 (10%) | 基准 |
| Exp2-1 | 5000 (15%) | 更平稳的早期学习 |
| Exp2-2 | 6000 (18%) | 最温和的学习率增长 |

---

## 实验 3: 数据采样策略
**目标**: 通过调整训练数据的类别比例来改善学习

### 实施位置

修改 `data_loader.py` 中的数据加载逻辑。

### 方案 A: 简单过采样

```python
# 在 data_loader.py 的数据加载部分添加

def balance_dataset(X, y, strategy='oversample', ratio=0.8):
    """
    平衡数据集

    Args:
        X: 特征数据
        y: 标签
        strategy: 'oversample', 'undersample', 或 'combined'
        ratio: 采样比例 (少数类 / 多数类)
    """
    from collections import Counter

    class_counts = Counter(y)
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    if strategy == 'oversample':
        # 过采样少数类
        target_size = int(len(class_0_idx) * ratio)
        current_size = len(class_1_idx)

        if current_size < target_size:
            oversample_idx = np.random.choice(
                class_1_idx,
                size=target_size - current_size,
                replace=True
            )
            class_1_idx = np.concatenate([class_1_idx, oversample_idx])

    balanced_idx = np.concatenate([class_0_idx, class_1_idx])
    np.random.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx]
```

### 方案 B: 使用 imbalanced-learn

```bash
# 先安装库
pip install imbalanced-learn

# 在 train.py 中使用
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 在准备训练数据时
smote_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('undersample', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
])

# 仅在训练集上应用
X_train_balanced, y_train_balanced = smote_pipeline.fit_resample(X_train, y_train)
```

### 实验配置

| 采样策略 | 过采样比例 | 欠采样比例 | 描述 |
|---------|----------|----------|------|
| 无采样 | - | - | 基准配置 |
| 简单过采样 | 0.8 | - | 将少数类增加到 80% |
| SMOTE + 欠采样 | 0.5 → 0.8 | 0.8 | SMOTE 生成样本后欠采样 |

---

## 实验 4: 综合优化
**目标**: 结合最有效的多个改进

### 推荐组合方案

```python
# 基于前三个实验的结果，选择最优配置组合

# train.py 第 128 行
criterion = FocalLoss(
    alpha=[1.2, 3.0],  # 来自实验 1
    gamma=2.5,
    label_smoothing=0.1
)

# train.py 中的学习率调度
warmup_steps = 5000  # 来自实验 2

# data_loader.py 中的采样
# 使用 SMOTE + UnderSampler（来自实验 3）
```

### 预期结果

| 指标 | 单个改进 | 综合改进 | 目标 |
|------|--------|--------|------|
| Macro F1 | +0.01-0.02 | +0.03-0.05 | **0.58-0.60** |
| Class 1 F1 | 更平衡 | 更强 | >0.55 |
| 收敛速度 | 略快 | 明显快 | <15 epochs |

---

## 执行记录模板

### 实验执行日志

```
=====================================
实验名称: [填写]
执行日期: 2025-12-27
主要改进: [填写修改的内容]

配置:
  - Focal Loss Alpha: [填写]
  - Focal Loss Gamma: [填写]
  - Label Smoothing: [填写]
  - 预热步数: [填写]
  - 采样策略: [填写]

结果:
  - 最终 Macro F1: [填写]
  - 最优轮次: [填写]
  - Class 0 F1: [填写]
  - Class 1 F1: [填写]
  - 最终 Accuracy: [填写]

观察:
  [填写训练过程中的观察，如收敛速度、预测分布等]

后续建议:
  [填写基于这个实验的后续改进方向]

Log 文件: [填写日志文件名]
=====================================
```

### 快速查看日志的关键指标

```bash
# 查看最终的 Macro F1
grep "f1_macro_cli" training_model*.log | tail -5

# 查看每个 epoch 的 Macro F1 变化
grep "f1_macro_cli" training_model*.log | head -20

# 查看预测分布变化
grep "Prediction distribution" training_model*.log

# 查看梯度信息
grep "gradient" training_model*.log
```

---

## 推荐实验顺序和时间估计

| 顺序 | 实验 | 预期耗时 | 优先级 |
|------|------|--------|-------|
| 1 | Focal Loss 权重优化 (4 个配置) | 2-3 小时 | ⭐⭐⭐⭐⭐ |
| 2 | 学习率预热调整 (2 个配置) | 1-2 小时 | ⭐⭐⭐⭐ |
| 3 | 数据采样策略 (2-3 个方案) | 2-3 小时 | ⭐⭐⭐⭐ |
| 4 | 综合优化 (选最佳组合) | 1-2 小时 | ⭐⭐⭐ |

**总耗时**: 6-10 小时（可并行执行）

---

## 快速参考：常见改进效果

基于文献和经验，以下改进在不平衡数据上的典型效果：

| 改进方法 | 预期 Macro F1 提升 | 适用场景 |
|---------|------------------|--------|
| 调整类别权重 | +0.01-0.03 | 权重不当 |
| 增加预热步数 | +0.005-0.015 | 早期学习不稳定 |
| 数据过采样 | +0.02-0.04 | 样本不足 |
| SMOTE 合成 | +0.02-0.05 | 需要生成新样本 |
| 多策略结合 | +0.03-0.08 | 综合优化 |

**当前情况**: 适合尝试前三个改进的组合

---

## 注意事项

1. **日志命名规范**: `training_model[N]_exp[实验号]_config[配置号].log`
2. **配置记录**: 每次实验都在日志开头或日志文件名中记录确切的配置
3. **数据一致性**: 确保验证/测试集始终保持一致，不进行采样
4. **梯度监控**: 如果发现梯度爆炸，立即降低权重比例
5. **随机种子**: 保持 `seed=42` 以确保可重现性

