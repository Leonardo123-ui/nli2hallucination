# LLM 幻觉检测推理系统

使用 llama3 和 qwen3 等大语言模型对幻觉检测数据进行推理，支持多种部署方式。

## 📁 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `llm_inference.py` | 🎯 **主推理脚本** - 支持Ollama、HuggingFace、API三种部署方式 |
| `config_examples.py` | 📋 配置示例 - 8个使用示例和配置模板 |
| `compare_models.py` | 📊 对比分析工具 - 对比BERT和LLM的推理结果 |

### 文档文件

| 文件 | 说明 |
|------|------|
| `QUICK_START.md` | ⚡ 快速开始指南（5分钟上手） |
| `LLM_INFERENCE_GUIDE.md` | 📖 详细使用指南（完整功能说明） |
| `requirements_llm.txt` | 📦 Python依赖列表 |
| `README.md` | 📄 本文件 |

## 🚀 快速开始

### 最简单的方式：使用Ollama（推荐）

```bash
# 1. 安装Ollama
curl https://ollama.ai/install.sh | sh

# 2. 启动服务（一个终端）
ollama serve

# 3. 拉取模型（另一个终端）
ollama pull llama3
ollama pull qwen:7b

# 4. 安装依赖
pip install -r requirements_llm.txt

# 5. 运行推理
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --sample_size 50
```

**完成！** 结果保存在 `./llm_results/` 中

## 📝 常见使用命令

### 使用不同模型

```bash
# llama3（英文）
python llm_inference.py --model_name llama3 --sample_size 100

# qwen 7B（中文）
python llm_inference.py --model_name qwen:7b --use_zh_prompt --sample_size 100

# qwen 14B（中文，更好的质量）
python llm_inference.py --model_name qwen:14b --use_zh_prompt --sample_size 100
```

### 对比分析

```bash
# 对比多个模型
python config_examples.py compare

# 对比BERT和LLM
python compare_models.py \
  --bert_dir ./test_results \
  --llm_dir ./llm_results
```

### 使用全部数据

```bash
# 使用所有900个测试样本
python llm_inference.py \
  --model_name llama3 \
  --sample_size None
```

## 📊 支持的模型

### Ollama（推荐，最简单）
- **llama3**：Meta的最新模型，性能好
- **qwen:7b**：阿里云QWen 7B，适合中文
- **qwen:14b**：更大的QWen，更好的性能

### HuggingFace（高性能）
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-70b-chat-hf
- THUDM/chatglm-6b
- QwenLM/Qwen-7B-Chat

### API（无硬件要求）
- 阿里云DashScope（Qwen系列）
- OpenAI / Together AI（Llama系列）

## 📈 推理结果

每次推理会生成以下输出文件：

```
./llm_results/
├── llm_results.json              # 评估指标总结
├── llm_detailed_predictions.xlsx # 详细预测结果
└── [可视化图表]                   # 可选的图表
```

### 输出示例

```json
{
  "model_name": "llama3",
  "test_size": 900,
  "valid_predictions": 890,
  "detailed_metrics": {
    "accuracy": 0.7651,
    "macro_f1": 0.6178,
    "hallucination": {
      "precision": 0.5145,
      "recall": 0.4234,
      "f1_score": 0.4646
    }
  }
}
```

## 🔧 三种部署方式对比

| 方式 | 优点 | 缺点 | 推荐 |
|------|------|------|------|
| **Ollama** | 最简单、自动优化内存 | 需要GPU | ✅ 最好选择 |
| **HuggingFace** | 完全控制、高性能 | 需要管理依赖 | 高级用户 |
| **API** | 无硬件要求、最强模型 | 需要API密钥、有费用 | 快速验证 |

## 📚 使用示例

### 示例1：快速测试
```bash
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --sample_size 50 \
  --output_dir ./results/quick_test
```

### 示例2：完整评估
```bash
python llm_inference.py \
  --model_name qwen:7b \
  --deploy_type ollama \
  --use_zh_prompt \
  --output_dir ./results/qwen_full
```

### 示例3：对比多个模型
```bash
# 使用配置示例脚本
python config_examples.py compare

# 生成对比报告
python config_examples.py report
```

### 示例4：对比BERT和LLM
```bash
python compare_models.py \
  --bert_dir ./test_results \
  --llm_dir ./llm_results \
  --output_dir ./comparison_results
```

## ⚙️ 命令行参数详解

### 模型配置
- `--model_name`: 模型名称（llama3, qwen:7b等）
- `--deploy_type`: 部署方式（ollama, huggingface, api）
- `--model_path`: HuggingFace/本地模型路径

### 数据配置
- `--data_path`: 测试数据文件路径
- `--sample_size`: 抽样大小（None为全部）

### 推理配置
- `--output_dir`: 结果保存目录
- `--use_zh_prompt`: 使用中文提示词
- `--temperature`: 采样温度（0为确定性）
- `--max_tokens`: 最大生成token数

## 🔍 结果分析

### Python分析示例

```python
import json
import pandas as pd

# 加载结果
with open('./llm_results/llm_results.json') as f:
    metrics = json.load(f)

# 打印关键指标
print(f"准确率: {metrics['detailed_metrics']['accuracy']:.2%}")
print(f"幻觉F1: {metrics['detailed_metrics']['hallucination']['f1_score']:.4f}")

# 加载详细预测
df = pd.read_excel('./llm_results/llm_detailed_predictions.xlsx')

# 统计错误分布
errors = df[df['correct_prediction'] == False]
print(f"错误率: {len(errors)/len(df):.2%}")
```

## 📖 更多文档

- **快速开始**: `QUICK_START.md` - 5分钟上手
- **完整指南**: `LLM_INFERENCE_GUIDE.md` - 详细功能说明
- **配置示例**: `config_examples.py` - 8个使用示例
- **对比工具**: `compare_models.py` - BERT vs LLM对比

## 🛠️ 故障排除

### 连接Ollama失败
```bash
# 确保Ollama正在运行
ollama serve

# 检查可用模型
ollama list
```

### 显存不足
- 使用Ollama（自动优化）
- 选择更小的模型（7B vs 14B）
- 减少样本数（--sample_size）

### 推理速度慢
- 使用GPU（检查torch.cuda.is_available()）
- 选择更小的模型
- 使用API方式

## 💡 提示

1. **首次运行**：使用--sample_size 50进行快速测试
2. **对比分析**：分别运行BERT和LLM，然后用compare_models.py对比
3. **Prompt优化**：修改HALLUCINATION_PROMPT_*来改进结果
4. **模型选择**：根据显存选择合适的模型大小

## 📞 支持

如有问题，请查看：
1. `QUICK_START.md` 中的常见问题
2. `LLM_INFERENCE_GUIDE.md` 中的详细说明
3. 命令行帮助：`python llm_inference.py --help`

## 📄 许可证

本项目遵循原项目的许可证。

---

**开始使用**: [快速开始指南](./QUICK_START.md)
