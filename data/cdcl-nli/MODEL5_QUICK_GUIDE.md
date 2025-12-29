# 快速行动指南 - Model 5: 关系类型优化

**执行时间**: 10 分钟修改 + 1.5 小时训练
**预期改进**: 0.5611 → 0.57-0.58

---

## 核心想法

当前模型使用 **20 个关系类型**，导致：
- ❌ 模型参数过多 (每个关系都要单独的 GAT)
- ❌ 某些关系（如 "span"）对分类无用
- ❌ 数据稀疏（900 个样本分散在 20 个关系上）

**解决方案**: 使用已经定义好的 **9 个关键关系** (`rel_names_short`)
- ✅ 减少参数约 55%
- ✅ 专注于重要的语篇关系
- ✅ 降低过拟合风险

---

## 两个修改点

### 修改 1: train.py 第 496 行

**当前代码**:
```python
model.merge_graphs(g_p1, g_p2, lc, rel_names_long)
```

**改为**:
```python
model.merge_graphs(g_p1, g_p2, lc, rel_names_short)
```

### 修改 2: train.py 第 718 行

**当前代码**:
```python
model = ExplainableHeteroClassifier(
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    rel_names=rel_names_long,  # ← 这里
)
```

**改为**:
```python
model = ExplainableHeteroClassifier(
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    n_classes=n_classes,
    rel_names=rel_names_short,  # ← 改成 short
)
```

---

## 执行步骤

### Step 1: 备份原文件
```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli
cp train.py train.py.backup_before_model5
```

### Step 2: 打开编辑器修改
```bash
nano train.py
```

**找到第 496 行**:
- 按 `Ctrl+G` 跳转到行号
- 输入 496
- 找到 `rel_names_long` 改为 `rel_names_short`

**找到第 718 行**:
- 按 `Ctrl+G` 跳转到行号
- 输入 718
- 找到 `rel_names_long` 改为 `rel_names_short`

**保存**:
- `Ctrl+O`, `Enter`, `Ctrl+X`

### Step 3: 验证修改
```bash
# 确认两处都改了
grep -n "rel_names" train.py | grep -E "496|718"

# 输出应该包含：
# 496: ...rel_names_short...
# 718: ...rel_names=rel_names_short...
```

### Step 4: 启动训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py > training_model5.log 2>&1 &
echo "Training started, log: training_model5.log"
```

### Step 5: 监控进度
```bash
# 实时查看
tail -f training_model5.log

# 或每 30 秒检查一次
watch -n 30 'grep "f1_macro_cli" training_model5.log | tail -3'
```

### Step 6: 对比结果 (训练完成后)
```bash
echo "=== 结果对比 ==="
echo "Model 3 (baseline with alpha=[1.0, 3.4]):"
grep -o 'f1_macro_cli: 0\.[0-9]*' training_model3.log | tail -1

echo "Model 5 (rel_names_short):"
grep -o 'f1_macro_cli: 0\.[0-9]*' training_model5.log | tail -1

echo ""
echo "改进:"
python3 << 'PYEOF'
import re
with open('training_model3.log') as f:
    m3 = float(re.findall(r'f1_macro_cli: (0\.\d+)', f.read())[-1])
with open('training_model5.log') as f:
    m5 = float(re.findall(r'f1_macro_cli: (0\.\d+)', f.read())[-1])
print(f"Model 5 vs Model 3: {m5 - m3:+.4f}")
PYEOF
```

---

## 预期结果

| 指标 | Model 3 | Model 5 | 变化 |
|------|---------|---------|------|
| Macro F1 | 0.5611 | **0.57-0.58** | ✅ +0.01-0.02 |
| Precision | 0.5597 | ? | ? |
| Recall | 0.5631 | ? | ? |
| 模型参数 | 20 rel | **9 rel** | -55% |

---

## 如果改进成功 (>0.005)

立即进行 **Model 6: 网络深度增加**

```
修改 build_base_graph_extract.py 中 RGAT 类的 conv2:

当前:
    num_heads=1,  # ← 只有 1 个头

改为:
    num_heads=4,  # ← 增加到 4 个头

这样第二层就能利用多头注意力的优势
```

---

## 如果效果不如预期 (<0.005)

回退到 Model 3，尝试其他方向：

```bash
# 恢复原文件
cp train.py.backup_before_model5 train.py

# 尝试混合方案：关键关系 + 更多关系
# 定义一个 15 个关系的中间列表
rel_names_medium = [
    "Temporal", "TextualOrganization", "Joint", "Topic-Comment",
    "Comparison", "Condition", "Contrast", "Evaluation", "Topic-Change",
    "Summary", "Attribution", "Cause", "Background", "Elaboration",
    "Explanation", "lexical"
]

# 用这个替代 rel_names_short 或 rel_names_long
```

---

## 关键代码位置快速查找

```bash
# 找到第一处修改点
grep -n "merge_graphs.*rel_names" train.py

# 找到第二处修改点
grep -n "rel_names=rel_names" train.py

# 查看两个列表的定义
grep -n "rel_names_long\|rel_names_short" train.py | head -40
```

---

## 常见问题

**Q: 只改这两行真的有效果吗？**
A: 是的，因为这影响到：
   1. 图的构建方式 (merge_graphs)
   2. 模型的初始化 (ExplainableHeteroClassifier)
   两处都改才能确保一致性

**Q: 改了之后要重新训练吗？**
A: 是的，需要从头开始训练。模型结构改变了。

**Q: 如果 Macro F1 下降怎么办？**
A: 这说明短列表可能丢失了重要关系。可以：
   - 尝试添加回某些关系（如 "Attribution", "Enablement"）
   - 定义一个 15-18 个关系的中间列表
   - 回到 Model 3 + 网络改进的方向

---

## 下一步（如果成功）

```
Model 5 成功 (改进 > 0.005)
  ↓
Model 6: 改进注意力头数 (5 分钟修改 + 1.5h 训练)
  ↓
Model 7: 添加网络深度 (30 分钟修改 + 2h 训练)
  ↓
最终目标: 0.59-0.60+
```

---

**现在就开始修改吧！预计 1.5 小时后见到结果。** 🚀

