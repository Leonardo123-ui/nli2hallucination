# 快速开始指南 - 立即可执行的改进

## 5 分钟快速改进 (推荐立即实施)

### 修改 1: 调整 Focal Loss 权重

**位置**: `/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/train.py` 第 128 行

**当前代码**:
```python
criterion = FocalLoss(alpha=[0.5, 2.0], gamma=2.0, label_smoothing=0.1)
```

**修改为** (选择一个尝试):
```python
# 推荐方案 1: 基于数据比例
criterion = FocalLoss(alpha=[1.0, 3.4], gamma=2.0, label_smoothing=0.1)

# 或方案 2: 稍微缓和
criterion = FocalLoss(alpha=[1.2, 3.0], gamma=2.5, label_smoothing=0.1)

# 或方案 3: 最激进
criterion = FocalLoss(alpha=[1.5, 2.5], gamma=2.0, label_smoothing=0.15)
```

**执行**:
```bash
# 运行训练（使用修改后的代码）
cd /mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli
CUDA_VISIBLE_DEVICES=0 python train.py > training_model3.log 2>&1
```

**预期效果**: Macro F1 提升 0.01-0.03 (从 0.5494 到 0.56-0.58)

**耗时**: ~1.5 小时

---

### 修改 2: 增加学习率预热步数 (可选同步)

**位置**: `/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/train.py`
(找到 `get_cosine_schedule_with_warmup` 的位置)

**寻找代码**:
```bash
grep -n "warmup_steps\|get_cosine_schedule" train.py
```

**修改方案**:
```python
# 当前可能是
warmup_steps = 3360  # 假设，需要检查实际值

# 改为
warmup_steps = 5000  # 增加预热步数
```

**预期效果**: 更平稳的早期学习，Macro F1 +0.005-0.015

---

## 30 分钟完整改进流程

### Step 1: 备份原文件
```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli
cp train.py train.py.backup
```

### Step 2: 修改配置
```bash
# 使用文本编辑器打开文件
nano train.py

# 找到第 128 行，修改为:
# criterion = FocalLoss(alpha=[1.0, 3.4], gamma=2.0, label_smoothing=0.1)

# 保存 (Ctrl+O, Enter, Ctrl+X)
```

### Step 3: 运行训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py > training_model3.log 2>&1 &
echo "训练已在后台启动，输出到 training_model3.log"
```

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py > training_model3.log 2>&1 &echo $!

### Step 4: 监控进度
```bash
# 实时查看日志（在另一个终端）
tail -f training_model3.log

# 或者定时检查关键指标
watch -n 60 'grep "f1_macro_cli" training_model3.log | tail -5'
```

### Step 5: 对比结果
```bash
# 训练完成后，对比 Macro F1
echo "Model 1 (原始):"
grep "f1_macro_cli:" training_model1.log | tail -1

echo "Model 3 (新增权重):"
grep "f1_macro_cli:" training_model3.log | tail -1
```

---

## 关键代码位置速查

### 需要修改的配置点

```python
# ==================== train.py ====================

# 第 128 行: Focal Loss 权重
criterion = FocalLoss(alpha=[0.5, 2.0], gamma=2.0, label_smoothing=0.1)
                            ↑ 改这里

# 搜索 "warmup_steps" 或 "get_cosine_schedule_with_warmup"
# 找到学习率调度，修改预热步数
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=3360,  # ← 改这个数字
    num_training_steps=total_steps
)

# ==================== 可选: data_loader.py ====================
# 如果要实施数据采样，在数据加载处添加平衡逻辑
```

---

## 结果快速判断标准

### 好的改进信号 ✓
- [ ] Macro F1 在第一个 epoch 就 > 0.40 (不是 0.43)
- [ ] 第 5 个 epoch 时预测分布已经有一定比例的 Class 1
- [ ] Macro F1 曲线平滑上升，无长期平坦区间
- [ ] 最终 Macro F1 > 0.55

### 需要调整的信号 ⚠️
- [ ] Macro F1 初期仍然平坦（表示权重不够）
- [ ] 前 10 个 epoch 无明显进度
- [ ] 梯度信息显示异常（如 NaN）

### 失败信号 ❌
- [ ] Macro F1 下降到 < 0.50
- [ ] 模型崩溃（Loss 为 NaN）
- [ ] 完全无法预测某个类别

---

## 常见问题排查

### Q1: 修改了权重后 Macro F1 反而下降？

**可能原因**:
1. 权重设置过极端（如 [5.0, 1.0]）
2. 梯度爆炸导致不稳定
3. 初始化不兼容

**解决方案**:
```bash
# 检查梯度是否异常
grep -i "gradient\|nan\|inf" training_model3.log | head -20

# 尝试更温和的权重
alpha=[1.2, 2.5]  # 而不是 [1.0, 3.4]

# 或者降低学习率
base_lr = 1e-5  # 原来的一半
```

### Q2: 训练过程中出现 NaN 损失？

**原因**: 权重过极端导致梯度爆炸

**解决方案**:
```python
# 添加梯度裁剪（如果还没有）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 或者更温和的权重
criterion = FocalLoss(alpha=[1.0, 2.5], gamma=2.0)
```

### Q3: 无法看到权重 Class 1 的提升？

**原因**: 权重仍然不足

**解决方案**:
```python
# 尝试更激进的权重
alpha=[0.8, 4.0]  # 5:1 的比例

# 或同时增加 gamma
gamma=3.0  # 更强的难样本关注

# 或同时调整学习率
warmup_steps = 6000  # 更长的预热
```

---

## 实验对比 Python 脚本

### 快速对比脚本
```python
#!/usr/bin/env python3
"""快速对比多个模型的 Macro F1"""

import re
import sys

def extract_metrics(log_file):
    """从日志文件中提取关键指标"""
    metrics = {
        'macro_f1_list': [],
        'epoch_list': [],
        'final_macro_f1': None,
        'final_accuracy': None,
    }

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取所有 Macro F1
    macro_f1_matches = re.findall(r'f1_macro_cli:\s+(0\.\d+)', content)
    if macro_f1_matches:
        metrics['macro_f1_list'] = [float(x) for x in macro_f1_matches]
        metrics['final_macro_f1'] = metrics['macro_f1_list'][-1]

    # 提取最后的 Accuracy
    accuracy_matches = re.findall(r'accuracy:\s+(0\.\d+)', content)
    if accuracy_matches:
        metrics['final_accuracy'] = float(accuracy_matches[-1])

    return metrics

def compare_models(log_files):
    """比较多个模型"""
    print("=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    print(f"{'Model':<30} | {'Final Macro F1':<15} | {'Accuracy':<10} | {'Improvement':<12}")
    print("-" * 80)

    baseline_f1 = None

    for log_file in log_files:
        try:
            metrics = extract_metrics(log_file)
            final_f1 = metrics['final_macro_f1']
            accuracy = metrics['final_accuracy']

            if baseline_f1 is None:
                baseline_f1 = final_f1
                improvement = "Baseline"
            else:
                improvement = f"+{(final_f1 - baseline_f1):.4f}"

            print(f"{log_file:<30} | {final_f1:<15.4f} | {accuracy:<10.4f} | {improvement:<12}")
        except Exception as e:
            print(f"{log_file:<30} | Error: {e}")

    print("=" * 80)

if __name__ == "__main__":
    # 使用方式
    if len(sys.argv) > 1:
        log_files = sys.argv[1:]
        compare_models(log_files)
    else:
        # 默认对比
        log_files = [
            'training_model1.log',
            'training_model3.log',  # 新实验
        ]
        compare_models(log_files)
```

**使用方法**:
```bash
# 保存为 compare_models.py
python compare_models.py training_model1.log training_model3.log

# 或
python compare_models.py training_model*.log
```

---

## 文件清单

我为你创建了以下文档，都在 `/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/` 目录:

| 文件名 | 说明 | 何时阅读 |
|-------|------|---------|
| **IMPROVEMENT_PLAN.md** | 详细的改进方案（7 个部分） | 第一次，了解全面策略 |
| **ANALYSIS_REPORT.md** | 对两个模型的深层分析 | 理解问题根源 |
| **EXPERIMENT_TEMPLATE.md** | 实验配置和执行模板 | 执行实验时参考 |
| **QUICK_START.md** | 本文件，快速开始指南 | 立即开始实施 |

---

## 推荐执行时间表

### 今天 (2025-12-27)
- [ ] 读完本快速开始指南 (5 分钟)
- [ ] 修改 Focal Loss 权重 (2 分钟)
- [ ] 启动训练（预计 1.5 小时）

### 训练期间
- [ ] 查看 ANALYSIS_REPORT.md 理解问题根源
- [ ] 准备后续的实验配置

### 训练完成后
- [ ] 对比 Model 3 vs Model 1 的结果
- [ ] 如果有提升 > 0.01，进行网格搜索
- [ ] 记录在 EXPERIMENT_TEMPLATE.md 中

### 本周
- [ ] 完成 4 个配置的完整实验
- [ ] 汇总最佳实验结果

---

## 紧急救援 (如果出错)

### 训练崩溃怎么办？

```bash
# 1. 停止训练
pkill -f "python train.py"

# 2. 恢复备份
cp train.py.backup train.py

# 3. 使用更温和的配置
# 修改 alpha=[1.2, 2.5] （更温和）

# 4. 重新启动
CUDA_VISIBLE_DEVICES=0 python train.py > training_model3_retry.log 2>&1
```

### Macro F1 没有改善怎么办？

```bash
# 1. 检查是否真的修改了
grep "FocalLoss" train.py

# 2. 尝试更激进的权重
alpha=[0.8, 4.0]  # 更大的权重比

# 3. 同时增加预热
warmup_steps = 6000

# 4. 检查梯度是否爆炸
grep -i "gradient" training_model3.log
```

### 想快速测试不同配置？

```bash
# 创建配置变体脚本
for alpha in "[1.0, 3.4]" "[1.2, 3.0]" "[1.5, 2.5]"; do
    sed -i "s/alpha=\[.*\]/alpha=$alpha/" train.py
    echo "Testing alpha=$alpha"
    CUDA_VISIBLE_DEVICES=0 python train.py > training_model3_$alpha.log 2>&1
done
```

---

## 最后的建议

1. **立即开始**: 不要过度计划，先修改权重运行一次看效果
2. **小步迭代**: 每次只改一个参数，观察效果
3. **记录一切**: 每个实验都用不同的 log 文件名
4. **耐心等待**: 训练需要 1.5 小时，这是正常的
5. **相信数据**: 如果 Macro F1 提升了，就值得继续尝试

---

**现在就开始吧！祝改进顺利！** 🚀
