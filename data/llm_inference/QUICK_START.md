# 🚀 快速开始指南

本指南帮助你快速开始使用LLM进行幻觉检测推理。

## 5分钟快速开始

### 前提条件
- Python 3.8+
- 足够的磁盘空间（模型需要5-10GB）

### 方法A: 最简单 - 使用Ollama（推荐）

#### 1. 安装Ollama
```bash
# 访问 https://ollama.ai 下载安装，或使用命令：
curl https://ollama.ai/install.sh | sh
```

#### 2. 启动Ollama服务
```bash
ollama serve
# 在另一个终端继续...
```

#### 3. 拉取模型（在新终端中）
```bash
# 拉取llama3（约4GB）
ollama pull llama3

# 拉取qwen（约3.5GB）
ollama pull qwen:7b

# 验证模型已安装
ollama list
```

#### 4. 安装依赖
```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/bert-classifier
pip install torch transformers pandas numpy scikit-learn tqdm requests
```

#### 5. 运行推理
```bash
# 快速测试（50个样本）
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --sample_size 50 \
  --output_dir ./results/llama3_test

# 完整测试（900个样本）
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --output_dir ./results/llama3_full
```

**完成！** 结果会保存在 `./results/llama3_full/` 中

---

### 方法B: 快速 - 使用HuggingFace模型

#### 1. 下载模型
```bash
# 如果有HuggingFace账户，这样下载最快
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama2

# 或指定本地llama路径
# 如果你已经有本地模型，跳过这步
```

#### 2. 运行推理
```bash
python llm_inference.py \
  --model_name llama2 \
  --deploy_type huggingface \
  --model_path ./models/llama2 \
  --sample_size 50
```

---

### 方法C: 无需下载 - 使用API

#### 阿里云Qwen API
```bash
# 设置API密钥
export DASHSCOPE_API_KEY="your_api_key"
pip install dashscope

# 运行推理
python llm_inference.py \
  --model_name qwen \
  --deploy_type api \
  --api_key $DASHSCOPE_API_KEY \
  --api_url https://dashscope.aliyuncs.com \
  --sample_size 50
```

---

## 常用命令速查表

```bash
# 使用llama3
python llm_inference.py --model_name llama3 --sample_size 100

# 使用qwen（中文）
python llm_inference.py --model_name qwen:7b --use_zh_prompt --sample_size 100

# 对比多个模型
python config_examples.py compare

# 生成对比报告（BERT vs LLM）
python compare_models.py

# 使用全部数据
python llm_inference.py --model_name llama3 --sample_size None
```

---

## 查看结果

### 快速查看
```bash
# 查看JSON结果摘要
cat ./results/llama3_test/llm_results.json

# 用Excel打开详细预测结果
open ./results/llama3_test/llm_detailed_predictions.xlsx
```

### Python查看
```python
import json
import pandas as pd

# 查看评估指标
with open('./results/llama3_test/llm_results.json') as f:
    results = json.load(f)
    print(results['detailed_metrics'])

# 查看详细预测
df = pd.read_excel('./results/llama3_test/llm_detailed_predictions.xlsx')
print(df.head())
```

---

## 性能对比

| 模型 | 显存 | 速度 | 质量 | 推荐用途 |
|-----|------|------|------|--------|
| llama3 7B | 4GB | 快 | 良好 | ✅ 快速测试 |
| qwen:7b | 4GB | 快 | 良好 | ✅ 快速测试 |
| llama3 70B | 40GB | 慢 | 优秀 | 精度要求高 |
| qwen:14b | 10GB | 中 | 优秀 | 平衡选择 |
| API | 0GB | 依赖网络 | 最优 | ✅ 无硬件限制 |

---

## 常见问题

### Q: 模型下载太慢怎么办？
**A:** 使用Ollama（自动优化）或API方式（无需下载）

### Q: 显存不足怎么办？
**A:** 使用API方式或Ollama的7B模型

### Q: 推理速度太慢怎么办？
**A:**
- 使用更小模型（7B vs 14B）
- 减少样本数（--sample_size）
- 检查是否用了GPU

### Q: 结果准确率低怎么办？
**A:**
- 尝试更大模型（14B or 70B）
- 调整Prompt（见高级用法）
- 数据可能确实困难

---

## 下一步

- 📖 [详细使用指南](./LLM_INFERENCE_GUIDE.md)
- 🔧 [配置示例](./config_examples.py)
- 📊 [对比分析工具](./compare_models.py)
- 💻 [完整源代码](./llm_inference.py)

---

## 需要帮助？

查看详细文档: `LLM_INFERENCE_GUIDE.md`

查看示例代码: `config_examples.py`

```bash
# 查看帮助信息
python llm_inference.py --help
python compare_models.py --help
```

---

**祝你使用愉快！🎉**
