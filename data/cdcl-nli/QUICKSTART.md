# 🚀 CDCL-NLI 幻觉检测 - 快速开始指南

## ⚡ 3步快速开始

### 第1步：检查环境

```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli

# 检查 Python 和依赖
python3 -c "import torch, transformers, pandas, numpy, nltk; print('✅ 所有依赖已安装')"

# 检查 GPU
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用')"
```

### 第2步：运行数据转换

```bash
# 转换幻觉检测数据为 NLI 格式
python3 convert_hallucination_data.py \
  --excel_path ../../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./data \
  --create_sample \
  --sample_size 100
```

**输出**:
- ✅ `data/hallucination_train.json` (4,758 样本)
- ✅ `data/hallucination_test.json` (900 样本)
- ✅ `data/hallucination_train_sample.json` (100 样本，用于测试)
- ✅ `data/hallucination_test_sample.json` (50 样本，用于测试)

### 第3步：处理数据

#### 选项A：使用一键脚本（推荐）

```bash
# 交互式运行，会引导你完成所有步骤
./run_pipeline.sh
```

#### 选项B：手动运行

```bash
# 完整处理（约 3-4 小时）
python3 arrange_hallucination_data.py
```

---

## 📋 快速测试（10分钟）

如果想快速验证流程，可以先用小样本测试：

```bash
# 1. 创建小样本数据（已在第2步完成）

# 2. 修改处理脚本使用小样本
# 编辑 arrange_hallucination_data.py，修改以下行：
# TRAIN_DATA_PATH = ".../hallucination_train_sample.json"
# TEST_DATA_PATH = ".../hallucination_test_sample.json"

# 3. 运行处理
python3 arrange_hallucination_data.py
```

**快速测试输出**:
- RST 结果: ~3分钟
- Embeddings: ~1分钟
- 词汇链: ~30秒
- 总计: ~5分钟

---

## 🔍 查看结果

### 方法1：命令行查看

```bash
# 查看训练集 RST 结果
head -1 ./data/train/rst_result.jsonl | python3 -m json.tool

# 查看文件大小
ls -lh ./data/train/*.npz
ls -lh ./data/graph_info/train/*.pkl
```

### 方法2：Python 查看

```python
import json
import torch
import pickle
import numpy as np

# 1. 查看 RST 结果
with open('./data/train/rst_result.jsonl', 'r') as f:
    rst_result = json.loads(f.readline())
    print("Premise 节点数:", len(rst_result['pre_node_string']))
    print("Hypothesis 节点数:", len(rst_result['hyp_node_string']))

# 2. 查看 Embeddings
embeddings = torch.load('./data/train/node_embeddings.npz')
print(f"\nEmbeddings 总数: {len(embeddings)}")
print(f"第一个样本的 premise 节点: {len(embeddings[0]['premise'])}")
print(f"Embedding 维度: {embeddings[0]['premise'][0][1].shape}")

# 3. 查看词汇链矩阵
with open('./data/graph_info/train/lexical_matrixes.pkl', 'rb') as f:
    matrices = pickle.load(f)
    print(f"\n词汇链矩阵数量: {len(matrices)}")
    print(f"第一个矩阵形状: {matrices[0].shape}")
    print(f"非零元素数: {np.count_nonzero(matrices[0])}")
```

---

## 📊 处理时间参考

| 数据集 | 样本数 | RST | Embeddings | 词汇链 | 总计 |
|--------|--------|-----|-----------|-------|------|
| **小样本(100)** | 100 | 3分钟 | 1分钟 | 30秒 | ~5分钟 |
| **测试集** | 900 | 20分钟 | 10分钟 | 5分钟 | ~35分钟 |
| **训练集** | 4,758 | 2小时 | 1小时 | 30分钟 | ~3.5小时 |

*基于 NVIDIA A100 40GB GPU

---

## ⚙️ 常用命令

### 数据转换

```bash
# 基本转换
python3 convert_hallucination_data.py

# 转换并创建小样本（推荐）
python3 convert_hallucination_data.py --create_sample --sample_size 100

# 自定义路径
python3 convert_hallucination_data.py \
  --excel_path /path/to/data.xlsx \
  --output_dir /path/to/output
```

### 数据处理

```bash
# 完整处理
python3 arrange_hallucination_data.py

# 使用一键脚本
./run_pipeline.sh
```

### 检查进度

```bash
# 查看 RST 结果数量
wc -l ./data/train/rst_result.jsonl

# 查看文件大小
du -h ./data/train/
du -h ./data/test/
du -h ./data/graph_info/
```

---

## ❌ 常见问题速查

### Q1: CUDA 内存不足

```python
# 在 arrange_hallucination_data.py 中修改批次大小
batch_size = 32  # 从 128 减少到 32
```

### Q2: 找不到 DM_RST 模块

```bash
# 检查路径
ls /mnt/nlp/yuanmengying/CDCL-NLI/data/DM_RST.py

# 添加到 Python 路径
export PYTHONPATH=$PYTHONPATH:/mnt/nlp/yuanmengying/CDCL-NLI
```

### Q3: ModernBERT 加载失败

```bash
# 检查模型
ls /mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large

# 测试加载
python3 -c "from transformers import AutoTokenizer, AutoModel; \
tokenizer = AutoTokenizer.from_pretrained('/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large'); \
print('✅ 模型加载成功')"
```

### Q4: 处理中断恢复

脚本会自动检测已处理的数据并跳过：

- RST 结果已存在 → 直接加载
- Embeddings 已存在 → 跳过生成
- 词汇链矩阵已存在 → 跳过计算

直接重新运行即可从中断处继续。

---

## 📁 输出文件说明

```
data/
├── hallucination_train.json              # 转换后的训练数据
├── hallucination_test.json               # 转换后的测试数据
│
├── train/
│   ├── rst_result.jsonl                  # ✅ RST 分析结果
│   ├── new_rst_result.jsonl              # ✅ 重写的 RST 结果（用于图构建）
│   └── node_embeddings.npz               # ✅ 节点 embeddings (ModernBERT)
│
├── test/
│   ├── rst_result.jsonl
│   ├── new_rst_result.jsonl
│   └── node_embeddings.npz
│
└── graph_info/
    ├── train/
    │   └── lexical_matrixes.pkl          # ✅ 词汇链矩阵 (训练集)
    └── test/
        └── lexical_matrixes.pkl          # ✅ 词汇链矩阵 (测试集)
```

---

## 🎯 下一步

处理完成后，你可以：

1. **查看数据**: 使用上面的 Python 代码查看生成的数据
2. **训练模型**: 使用生成的数据训练 CDCL-NLI 图神经网络模型
3. **调整参数**: 修改词汇链阈值、批次大小等参数重新处理

---

## 📖 详细文档

- **完整文档**: 查看 `README.md`
- **脚本说明**: 查看各脚本的注释

---

## 💡 建议

1. **首次使用**: 先用小样本（100个）快速测试
2. **正式处理**: 使用完整数据集，建议晚上运行
3. **检查结果**: 处理完成后检查文件大小和样本数量
4. **备份数据**: 处理结果较大，建议定期备份

---

**祝你使用愉快！** 🎉

有问题请查看 `README.md` 的常见问题部分。
