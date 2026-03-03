# LLM增强训练脚本改进说明

## 🎯 核心改进

### 1. ✅ 文本提取功能
新增 `extract_text_from_graph()` 函数：
- 从DGL图的节点中提取实际的前提(premise)和假设(hypothesis)文本
- 自动解码Base64编码的文本数据
- 过滤空文本，拼接多个节点的文本内容

### 2. ✅ Rationale缓存机制
新增 `RationaleCache` 类：
- 使用MD5哈希作为样本唯一标识 (premise + hypothesis)
- 自动持久化到 `cache/rationales/rationale_cache.json`
- 首次训练时调用LLM，后续epoch直接从缓存加载
- 每个epoch结束后自动保存缓存

**缓存统计示例**：
```
Batch rationale统计: LLM调用=5, 缓存命中=25
```

### 3. ✅ 改进的LLM调用
**之前的问题**：
```python
# ❌ LLM没有收到实际文本
user_input=f"样本 {sample_idx}: 分析关键的语义不一致之处"
```

**现在的方案**：
```python
# ✅ 传入实际的premise和hypothesis文本
user_input=(
    f"前提: {premise_text}\n"
    f"假设: {hypothesis_text}\n\n"
    f"请简要分析它们之间的关键语义关系..."
)
```

### 4. ✅ 训练/评估一致性
- **训练阶段**：使用LLM增强的图 + 缓存机制
- **评估阶段**：同样使用LLM增强的图（从缓存加载）
- 确保训练和测试数据分布一致

### 5. ✅ 显示F1指标
- 进度条实时显示F1分数而不是accuracy
- 更适合评估NLI任务的分类效果

---

## 📊 效率提升

### 第1个epoch：
- 需要调用 `159 batch × 30 samples = 4770次 LLM`
- 预计时间：~6小时

### 第2个epoch开始：
- **100% 缓存命中**，无需调用LLM
- 预计时间：~15-20分钟/epoch
- **效率提升 ~20倍**

---

## 🚀 使用方法

### 方法1：默认启用缓存（推荐）
```bash
python train_llm_augmented.py
```

### 方法2：禁用缓存（每次都调用LLM）
修改 `main()` 函数：
```python
trainer = LLMAugmentedTrainer(
    model=model,
    llm_config=llm_config,
    use_cache=False  # 禁用缓存
)
```

---

## 📁 文件结构

```
cdcl-nli/
├── train_llm_augmented.py          # 修改后的训练脚本
├── llm_graph_augmentation.py       # 图增强模块（未修改）
├── cache/
│   └── rationales/
│       └── rationale_cache.json    # 自动生成的缓存文件
└── logs/
    └── training_model_llm1.log     # 训练日志
```

---

## 🔍 验证缓存功能

### 检查缓存文件：
```bash
cat cache/rationales/rationale_cache.json | head -20
```

### 查看缓存大小：
```bash
du -h cache/rationales/rationale_cache.json
```

### 清空缓存重新生成：
```bash
rm -rf cache/rationales/
```

---

## ⚠️ 注意事项

1. **首次训练较慢**：第1个epoch需要生成所有rationale（~6小时）
2. **后续训练极快**：第2个epoch开始直接从缓存加载（~20分钟）
3. **缓存文件大小**：约 4770条 × 200字符 ≈ 2-5MB
4. **缓存失效场景**：
   - 修改了前提/假设的预处理方式
   - 更换了数据集
   - 需要重新生成时，删除 `cache/` 目录即可

---

## 📈 预期效果

### 训练稳定性提升：
- ✅ Loss曲线平滑，不再剧烈波动
- ✅ 每个epoch看到相同的增强特征
- ✅ 模型收敛更稳定

### 训练/测试一致性：
- ✅ 训练和评估都使用相同的LLM增强
- ✅ 评估指标更可信

### 效率大幅提升：
- ✅ 第2个epoch开始训练速度提升 ~20倍
- ✅ 总训练时间从 ~300小时 降至 ~20小时（50 epochs）

---

## 🐛 故障排除

### 问题1：缓存文件损坏
```bash
rm cache/rationales/rationale_cache.json
# 重新训练即可
```

### 问题2：LLM调用失败
- 检查日志中的错误信息
- 失败的样本会使用空rationale（不影响训练）
- 后续epoch会重试失败的样本

### 问题3：文本提取失败
- 查看日志：`图中没有text_encoded属性`
- 确认数据集正确加载
- 检查 `path_ini.py` 中的数据路径

---

## 📝 日志示例

```
[✓] Rationale缓存已启用，已有 0 条缓存
Epoch 0 Training:   1%|▏ | 1/159 [02:13<5:51:13] loss=0.701, f1=0.523
Batch rationale统计: LLM调用=30, 缓存命中=0
...
缓存已保存，共 4770 条记录

[✓] Rationale缓存已启用，已有 4770 条缓存
Epoch 1 Training:   1%|▏ | 1/159 [00:08<00:21] loss=0.659, f1=0.687
Batch rationale统计: LLM调用=0, 缓存命中=30
```

---

## 🎓 技术细节

### 样本哈希计算：
```python
content = f"{premise_text}|||{hypothesis_text}"
hash = hashlib.md5(content.encode('utf-8')).hexdigest()
```

### 缓存存储格式：
```json
{
  "a3f5d9e2...": "前提和假设存在明显的逻辑矛盾...",
  "b8c4e1f7...": "假设中缺少前提中的关键信息..."
}
```

---

## 🔮 未来优化方向

1. **并行LLM调用**：使用多线程/异步调用加速首次生成
2. **增量缓存**：每个batch结束后立即保存
3. **分布式缓存**：多GPU训练时共享缓存
4. **Rationale质量评估**：过滤低质量的LLM输出
