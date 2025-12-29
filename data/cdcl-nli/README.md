# CDCL-NLI 幻觉检测适配

将幻觉检测数据适配到 CDCL-NLI 模型，使用 RST (修辞结构树) 和图神经网络进行幻觉检测。

## 📋 项目概述

本项目将幻觉检测任务适配到 CDCL-NLI (Cross-Document Cross-Lingual NLI) 框架：

1. **输入数据**: 幻觉检测数据集 (context + output + label)
2. **处理流程**:
   - 将 context 作为 premise (前提)
   - 将 output 作为 hypothesis (假设)
   - 使用 DM-RST 模型提取修辞结构树
   - 使用 ModernBERT 生成节点 embeddings
   - 计算词汇链（lexical chains）矩阵
3. **输出**: 适用于 CDCL-NLI 图神经网络模型的数据

## 📂 项目结构

```
cdcl-nli/
├── convert_hallucination_data.py    # 数据格式转换脚本
├── arrange_hallucination_data.py    # 主数据处理脚本
├── run_pipeline.sh                   # 一键运行脚本
├── README.md                         # 本文件
│
├── data/                             # 数据目录
│   ├── hallucination_train.json     # 转换后的训练数据
│   ├── hallucination_test.json      # 转换后的测试数据
│   │
│   ├── train/                        # 训练集处理结果
│   │   ├── rst_result.jsonl         # RST 分析结果
│   │   ├── new_rst_result.jsonl     # 重写的 RST 结果
│   │   └── node_embeddings.npz      # 节点 embeddings
│   │
│   ├── test/                         # 测试集处理结果
│   │   ├── rst_result.jsonl
│   │   ├── new_rst_result.jsonl
│   │   └── node_embeddings.npz
│   │
│   └── graph_info/                   # 图结构信息
│       ├── train/
│       │   └── lexical_matrixes.pkl # 词汇链矩阵 (训练集)
│       └── test/
│           └── lexical_matrixes.pkl # 词汇链矩阵 (测试集)
│
└── DM_RST/                           # RST 模型模块 (软链接)
```

## 🚀 快速开始

### 前置要求

1. **Python 环境**:
   ```bash
   python >= 3.8
   CUDA 可用（推荐）
   ```

2. **依赖包**:
   ```bash
   pip install torch transformers pandas numpy nltk tqdm
   ```

3. **DM-RST 模型**:
   - 模型已在 `/mnt/nlp/yuanmengying/CDCL-NLI/data/DM_RST.py`
   - 脚本会自动引用

4. **ModernBERT 模型**:
   - 路径: `/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large`

### 运行步骤

#### 步骤 1: 转换数据格式

```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli

# 转换幻觉检测数据为 NLI 格式
python convert_hallucination_data.py \
  --excel_path ../../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./data \
  --create_sample
```

**输出**:
- `data/hallucination_train.json` (4,758 样本)
- `data/hallucination_test.json` (900 样本)
- `data/hallucination_train_sample.json` (100 样本，用于测试)
- `data/hallucination_test_sample.json` (50 样本，用于测试)

#### 步骤 2: 处理数据（RST + Embeddings + 词汇链）

```bash
# 完整处理（需要较长时间）
python arrange_hallucination_data.py
```

**处理时间估算**:
- 训练集 (4,758 样本): ~3-4 小时
- 测试集 (900 样本): ~30-40 分钟

**处理流程**:
1. 加载转换后的 JSON 数据
2. 使用 DM-RST 模型提取修辞结构树
3. 使用 ModernBERT 生成节点 embeddings
4. 计算词汇链矩阵（基于余弦相似度）

#### 步骤 3: 查看结果

```python
import json
import torch
import pickle

# 查看 RST 结果
with open('./data/train/rst_result.jsonl', 'r') as f:
    line = f.readline()
    rst_result = json.loads(line)
    print("RST 结果示例:")
    print(json.dumps(rst_result, indent=2))

# 查看 embeddings
embeddings = torch.load('./data/train/node_embeddings.npz')
print(f"\nEmbeddings 数量: {len(embeddings)}")
print(f"第一个样本的 premise 节点数: {len(embeddings[0]['premise'])}")

# 查看词汇链矩阵
with open('./data/graph_info/train/lexical_matrixes.pkl', 'rb') as f:
    matrices = pickle.load(f)
    print(f"\n词汇链矩阵数量: {len(matrices)}")
    print(f"第一个矩阵形状: {matrices[0].shape}")
```

## 📝 数据格式说明

### 输入数据（Excel）

| 列名 | 说明 | 示例 |
|------|------|------|
| id | 样本ID | `summary_train_0` |
| context | 上下文（长文本） | `Seventy years ago...` |
| output | 生成的摘要 | `The Anne Frank House...` |
| label | 标签（0=无幻觉, 1=有幻觉） | `0` |
| split | 数据集划分 | `train` / `test` |
| task_type | 任务类型 | `Summary` |

### 转换后数据（JSON）

```json
{
  "news1_origin": "context text...",  // 原始 context
  "news2_origin": "output text...",   // 原始 output
  "label": 0,                         // NLI 标签 (0=entailment, 2=contradiction)
  "original_label": 0,                // 原始幻觉标签 (0=无幻觉, 1=有幻觉)
  "id": "summary_train_0",
  "task_type": "Summary"
}
```

**标签映射**:
- 无幻觉 (0) → entailment (0) - output 与 context 一致
- 有幻觉 (1) → contradiction (2) - output 与 context 矛盾

### RST 结果（JSONL）

每行一个 JSON 对象，包含：

```json
{
  "pre_node_number": [...],      // premise 节点编号
  "pre_node_string": [...],      // premise 节点字符串
  "pre_node_relations": [...],   // premise 节点关系
  "pre_tree": [...],             // premise 树结构
  "pre_leaf_node": [...],        // premise 叶子节点
  "pre_parent_dict": {...},      // premise 父节点字典
  "hyp_node_number": [...],      // hypothesis 节点编号（类似）
  "hyp_node_string": [...],
  "hyp_node_relations": [...],
  "hyp_tree": [...],
  "hyp_leaf_node": [...],
  "hyp_parent_dict": {...}
}
```

### 节点 Embeddings（.npz）

```python
[
  {
    "premise": [
      (node_id, embedding_array, text_string),
      ...
    ],
    "hypothesis": [
      (node_id, embedding_array, text_string),
      ...
    ]
  },
  ...
]
```

### 词汇链矩阵（.pkl）

```python
[
  np.array([[0.0, 0.1, ...],  # premise 节点 0 与 hypothesis 各节点的相似度
            [0.2, 0.0, ...],  # premise 节点 1 与 hypothesis 各节点的相似度
            ...]),
  ...
]
```

## 🔧 脚本参数说明

### `convert_hallucination_data.py`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--excel_path` | Excel 数据文件路径 | `../../summary_nli_hallucination_dataset.xlsx` |
| `--output_dir` | 输出目录 | `./data` |
| `--create_sample` | 是否创建小样本数据 | `False` |
| `--sample_size` | 小样本大小 | `100` |

### `arrange_hallucination_data.py`

主要配置在脚本内部（`if __name__ == "__main__"` 部分）：

```python
MODEL_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large"
OVERALL_SAVE_DIR = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data"
GRAPH_INFOS_DIR = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/data/graph_info"
```

## 💡 使用建议

### 1. 快速测试

使用小样本数据快速验证流程：

```bash
# 创建小样本
python convert_hallucination_data.py --create_sample --sample_size 10

# 修改 arrange_hallucination_data.py 中的数据路径为样本路径
# TRAIN_DATA_PATH = ".../hallucination_train_sample.json"
# TEST_DATA_PATH = ".../hallucination_test_sample.json"

# 运行处理
python arrange_hallucination_data.py
```

### 2. 分步处理

如果内存不足，可以分步处理：

```python
# 1. 只运行 RST 分析
data_processor = HallucinationDataProcessor(True, OVERALL_SAVE_DIR, "train")
train_data, train_rst_result = load_all_data(
    data_processor, TRAIN_DATA_PATH, TRAIN_RST_RESULT_PATH
)

# 2. 运行 embedding 生成
embedder = ModernBERTEmbedder(MODEL_PATH, GRAPH_INFOS_DIR, "train", True)
# ... (后续步骤)
```

### 3. 调整批次大小

如果显存不足，减小批次大小：

```python
# 在 get_modernbert_embeddings_in_batches 中
batch_size = 64  # 从 128 减少到 64
```

### 4. 调整词汇链阈值

```python
# 在 find_lexical_chains 中
threshold = 0.7  # 从 0.8 降低到 0.7，会有更多词汇链连接
```

## 🐛 常见问题

### Q1: CUDA 内存不足

**解决方案**:
```python
# 减小批次大小
batch_size = 32  # 或更小

# 清理 GPU 缓存
import torch
torch.cuda.empty_cache()
```

### Q2: RST 模型无法加载

**解决方案**:
```bash
# 检查 DM-RST 路径
ls /mnt/nlp/yuanmengying/CDCL-NLI/data/DM_RST.py

# 确保 Python 路径正确
export PYTHONPATH=$PYTHONPATH:/mnt/nlp/yuanmengying/CDCL-NLI
```

### Q3: ModernBERT 模型加载失败

**解决方案**:
```bash
# 检查模型路径
ls /mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large

# 测试模型加载
python -c "from transformers import AutoTokenizer, AutoModel; \
tokenizer = AutoTokenizer.from_pretrained('/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large'); \
model = AutoModel.from_pretrained('/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large'); \
print('模型加载成功')"
```

### Q4: 处理速度太慢

**优化建议**:
1. 使用 GPU（确保 `torch.cuda.is_available()` 返回 True）
2. 增大批次大小（如果显存允许）
3. 使用小样本先测试
4. 考虑并行处理（分割数据集）

## 📊 性能参考

| 数据集 | 样本数 | RST 时间 | Embedding 时间 | 词汇链时间 | 总时间 |
|--------|--------|---------|--------------|----------|--------|
| 训练集 | 4,758 | ~2小时 | ~1小时 | ~30分钟 | ~3.5小时 |
| 测试集 | 900 | ~20分钟 | ~10分钟 | ~5分钟 | ~35分钟 |
| 小样本(100) | 100 | ~3分钟 | ~1分钟 | ~0.5分钟 | ~4.5分钟 |

*基于 NVIDIA A100 40GB GPU 的估算时间

## 🔗 相关资源

- [CDCL-NLI 原始项目](https://github.com/...)
- [DM-RST 论文](https://...)
- [ModernBERT 模型](https://huggingface.co/...)

## 📧 联系方式

有问题或建议？请联系项目维护者。

---

**最后更新**: 2024年
**版本**: 1.0
