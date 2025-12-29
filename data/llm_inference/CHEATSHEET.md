# 🎯 LLM推理 - 快速参考卡

## ⚡ 最快开始（5分钟）

```bash
# 1️⃣ 启动Ollama（一个终端）
ollama serve

# 2️⃣ 拉取模型（另一个终端）
ollama pull llama3
ollama pull qwen:7b

# 3️⃣ 快速推理
cd /mnt/nlp/yuanmengying/nli2hallucination/data/bert-classifier
python llm_inference.py --model_name llama3 --sample_size 50

# 4️⃣ 查看结果
cat ./llm_results/llm_results.json
```

---

## 📋 常用命令

### 基础推理

```bash
# llama3（英文，推荐）
python llm_inference.py --model_name llama3

# qwen 7B（中文）
python llm_inference.py --model_name qwen:7b --use_zh_prompt

# qwen 14B（更好的质量）
python llm_inference.py --model_name qwen:14b --use_zh_prompt
```

### 采样和限制

```bash
# 快速测试（50个样本）
python llm_inference.py --model_name llama3 --sample_size 50

# 中等测试（200个样本）
python llm_inference.py --model_name llama3 --sample_size 200

# 完整测试（所有900个样本）
python llm_inference.py --model_name llama3
```

### 对比分析

```bash
# 对比多个模型
python config_examples.py compare

# BERT vs LLM对比
python compare_models.py

# 生成对比报告
python config_examples.py report
```

### 环境检查

```bash
# 检查环境是否就绪
python check_environment.py
```

---

## 📂 输出位置

```bash
# 推理结果
./llm_results/
  └── llm_results.json              # 评估指标
  └── llm_detailed_predictions.xlsx # 详细预测

# 对比结果
./comparison_results/
  └── bert_llm_comparison.xlsx      # 对比结果
  └── disagreement_cases.xlsx       # 分歧案例
```

---

## 🔧 参数速查

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_name` | 模型名称 | `llama3`, `qwen:7b` |
| `--deploy_type` | 部署方式 | `ollama`, `huggingface` |
| `--sample_size` | 采样大小 | `50`, `100`, `None` |
| `--use_zh_prompt` | 中文提示 | (存在=True) |
| `--output_dir` | 输出目录 | `./results` |

---

## 🔍 查看结果

```python
# Python查看
import json, pandas as pd

# 查看指标
with open('./llm_results/llm_results.json') as f:
    metrics = json.load(f)['detailed_metrics']
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"幻觉F1: {metrics['hallucination']['f1_score']:.4f}")

# 查看详细预测
df = pd.read_excel('./llm_results/llm_detailed_predictions.xlsx')
print(df[['id', 'label', 'llm_prediction', 'correct_prediction']])
```

---

## ❌ 遇到问题？

| 问题 | 解决方案 |
|------|--------|
| Ollama连接失败 | `ollama serve` 启动服务 |
| 显存不足 | 用更小模型（7B）或减少样本 |
| 推理很慢 | 检查GPU：`python -c "import torch; print(torch.cuda.is_available())"` |
| 模型未找到 | `ollama list` 查看，`ollama pull llama3` 安装 |

---

## 📖 详细文档

| 文件 | 内容 |
|------|------|
| `QUICK_START.md` | 5分钟上手指南 |
| `LLM_INFERENCE_GUIDE.md` | 完整功能说明 |
| `config_examples.py` | 8个代码示例 |
| `PROJECT_SUMMARY.md` | 项目总体说明 |

---

## 💻 系统要求

- Python 3.8+
- 硬盘空间：5-10GB（用于模型）
- GPU显存：4GB+（使用Ollama）或 8GB+（HuggingFace）

---

## 🚀 推荐流程

```
Day 1: 快速验证
  └─ python llm_inference.py --sample_size 50

Day 2-3: 完整评估
  └─ python llm_inference.py

Day 4-5: 模型对比
  └─ python config_examples.py compare
  └─ python compare_models.py

Day 6+: 优化和集成
  └─ 调整Prompt或尝试更大模型
  └─ 集成到生产流程
```

---

## 🎉 立即开始！

```bash
# 检查环境
python check_environment.py

# 快速测试
python llm_inference.py --model_name llama3 --sample_size 50

# 查看帮助
python llm_inference.py --help
```

---

**更多帮助**: 查看 `QUICK_START.md` 或 `LLM_INFERENCE_GUIDE.md`
