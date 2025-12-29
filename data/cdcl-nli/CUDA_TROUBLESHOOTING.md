# 🔧 CUDA 错误故障排查指南

## ❌ 常见错误

### 错误1: CUDA out of memory (显存不足)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

1. **减小批次大小** (推荐)
   ```python
   # 在 arrange_hallucination_data.py 的 get_node_string_pair 方法中
   # 已修改为 batch_size=32（从128降低）

   # 如果还不够，可以进一步减小到16或8
   premise_embeddings = self.get_modernbert_embeddings_in_batches(
       premise_texts, batch_size=16  # 或 8
   )
   ```

2. **清理GPU缓存**
   ```bash
   # 运行前先清理
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

3. **检查GPU使用情况**
   ```bash
   nvidia-smi
   # 如果有其他程序占用GPU，先关闭
   ```

---

### 错误2: CUDA error: an illegal memory access was encountered

**症状**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**原因**: 通常是输入数据有问题或模型输出维度不匹配

**解决方案**:

✅ **已在脚本中修复**:
- 添加了错误处理和重试机制
- 使用 float16 减少显存占用
- 添加了 CUDA_LAUNCH_BLOCKING=1 用于调试
- 减小批次大小到32

**如果还有问题，尝试**:

1. **使用更小的批次**
   ```python
   batch_size=8  # 或更小
   ```

2. **检查输入数据**
   ```python
   # 查看是否有异常长的文本
   import json
   with open('./data/hallucination_train.json') as f:
       data = json.load(f)
       for item in data[:10]:
           print(f"Context length: {len(item['news1_origin'])}")
           print(f"Output length: {len(item['news2_origin'])}")
   ```

3. **重启Python进程**
   ```bash
   # 完全清理GPU
   pkill -9 python
   # 然后重新运行
   ```

---

### 错误3: ModernBERT 加载失败

**症状**:
```
OSError: Can't load tokenizer/model
```

**解决方案**:

1. **检查模型路径**
   ```bash
   ls -la /mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large
   ```

2. **测试模型加载**
   ```python
   from transformers import AutoTokenizer, AutoModel

   model_path = "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large"

   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModel.from_pretrained(model_path)

   print("✅ 模型加载成功")
   ```

---

## 🚀 优化建议

### 1. 逐步测试

```bash
# 第1步: 用10个样本测试
python3 -c "
import json
with open('./data/hallucination_train.json') as f:
    data = json.load(f)
with open('./data/hallucination_train_tiny.json', 'w') as f:
    json.dump(data[:10], f, indent=2)
"

# 第2步: 修改脚本使用 tiny 数据
# TRAIN_DATA_PATH = ".../hallucination_train_tiny.json"

# 第3步: 运行测试
python3 arrange_hallucination_data.py
```

### 2. 监控GPU使用

```bash
# 在另一个终端
watch -n 1 nvidia-smi
```

### 3. 使用小批次

当前脚本已优化为：
- ✅ batch_size=32 (从128降低)
- ✅ float16 精度 (减少50%显存)
- ✅ 自动错误重试
- ✅ 定期清理GPU缓存

---

## 📊 性能参考

| GPU | 显存 | 推荐批次大小 | 预期时间(4758样本) |
|-----|------|------------|------------------|
| A100 40GB | 40GB | 64-128 | ~1小时 |
| V100 32GB | 32GB | 32-64 | ~1.5小时 |
| RTX 3090 24GB | 24GB | 16-32 | ~2小时 |
| RTX 3080 10GB | 10GB | 8-16 | ~3小时 |

---

## 🔍 调试步骤

### 步骤1: 检查环境

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB' if torch.cuda.is_available() else 'N/A')
"
```

### 步骤2: 清理GPU

```bash
# 杀掉所有Python进程
pkill -9 python

# 清理GPU缓存
nvidia-smi --gpu-reset
```

### 步骤3: 重新运行

```bash
# 使用小批次
CUDA_VISIBLE_DEVICES=0 python3 arrange_hallucination_data.py
```

---

## 💡 最佳实践

### 1. 首次运行

```bash
# 使用小样本测试
python3 convert_hallucination_data.py --create_sample --sample_size 10

# 修改脚本使用sample数据
# 运行测试
python3 arrange_hallucination_data.py
```

### 2. 正式运行

```bash
# 监控GPU
watch -n 1 nvidia-smi &

# 后台运行
nohup python3 arrange_hallucination_data.py > process.log 2>&1 &

# 查看日志
tail -f process.log
```

### 3. 出错恢复

脚本会自动：
- ✅ 检测已处理的数据
- ✅ 跳过已完成的步骤
- ✅ 从中断处继续

直接重新运行即可：
```bash
python3 arrange_hallucination_data.py
```

---

## 📞 获取详细错误信息

如果仍有问题，运行以下命令获取详细信息：

```bash
# 设置详细调试
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 运行脚本
python3 arrange_hallucination_data.py 2>&1 | tee debug.log

# 查看错误详情
cat debug.log
```

---

## ✅ 当前脚本优化

已实施的优化：
- ✅ batch_size: 128 → 32
- ✅ 使用 float16 精度
- ✅ 添加错误处理和重试
- ✅ 定期清理GPU缓存
- ✅ 添加 CUDA_LAUNCH_BLOCKING
- ✅ 单个样本fallback机制

---

**问题解决后，可以正常使用小批次大小运行！**
