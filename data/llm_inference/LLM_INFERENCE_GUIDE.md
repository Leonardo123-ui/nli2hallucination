# LLM 幻觉检测推理指南

本文档介绍如何使用 `llm_inference.py` 脚本，通过 llama3 和 qwen3 等大语言模型对幻觉检测数据进行推理。

## 目录

1. [环境准备](#环境准备)
2. [部署方式](#部署方式)
3. [快速开始](#快速开始)
4. [详细用法](#详细用法)
5. [输出结果](#输出结果)
6. [对比分析](#对比分析)

---

## 环境准备

### 安装依赖包

```bash
pip install torch transformers pandas numpy scikit-learn tqdm requests

# 如果使用API方式，还需要安装对应的库
# 阿里云Qwen API
pip install dashscope

# OpenAI API（支持llama）
pip install openai
```

### 硬件要求

| 部署方式 | 显存要求 | CPU内存要求 |
|---------|---------|----------|
| Ollama（本地） | 4-8GB | 8GB+ |
| HuggingFace加载 | 8-16GB | 16GB+ |
| API调用 | 无 | 低 |

---

## 部署方式

### 方式1：使用Ollama（推荐，最简单）

#### 1.1 安装Ollama

```bash
# 访问官网下载安装
https://ollama.ai

# 或使用包管理器（Linux）
curl https://ollama.ai/install.sh | sh
```

#### 1.2 启动Ollama服务

```bash
# 默认在localhost:11434
ollama serve

# 或在后台运行
nohup ollama serve > ollama.log 2>&1 &
```

#### 1.3 拉取模型

```bash
# 拉取llama3
ollama pull llama3

# 拉取qwen（需要Ollama 0.1.23+）
ollama pull qwen:7b
ollama pull qwen:14b

# 查看已安装的模型
ollama list
```

#### 1.4 运行推理

```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/bert-classifier

# 使用llama3
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./llm_results/llama3 \
  --sample_size 50  # 先用50个样本测试

# 使用qwen
python llm_inference.py \
  --model_name qwen:7b \
  --deploy_type ollama \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./llm_results/qwen \
  --sample_size 50
```

---

### 方式2：使用HuggingFace直接加载

#### 2.1 下载模型

```bash
# 下载llama3（需要HuggingFace Access Token）
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama3

# 或使用Python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="your_hf_token")
model = AutoModelForCausalLM.from_pretrained(model_id, token="your_hf_token")

model.save_pretrained("./models/llama3")
tokenizer.save_pretrained("./models/llama3")
```

#### 2.2 运行推理

```bash
# 使用本地HuggingFace模型
python llm_inference.py \
  --model_name llama3 \
  --deploy_type huggingface \
  --model_path ./models/llama3 \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./llm_results/llama3 \
  --sample_size 100
```

---

### 方式3：使用API调用

#### 3.1 阿里云Qwen API

```bash
# 安装dashscope库
pip install dashscope

# 设置API密钥
export DASHSCOPE_API_KEY="your_api_key"

# 运行推理
python llm_inference.py \
  --model_name qwen \
  --deploy_type api \
  --api_url https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation \
  --api_key your_api_key \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./llm_results/qwen_api
```

#### 3.2 OpenAI兼容API（如Together、Replicate等）

```bash
# 安装openai库
pip install openai

# 运行推理
python llm_inference.py \
  --model_name llama-2-70b \
  --deploy_type api \
  --api_key your_api_key \
  --api_url https://api.together.xyz \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./llm_results/llama3_api
```

---

## 快速开始

### 最简单的开始方式：使用Ollama

```bash
# 第1步：安装并启动Ollama（一次性）
curl https://ollama.ai/install.sh | sh
ollama serve

# 第2步：在另一个终端拉取模型（一次性）
ollama pull llama3
ollama pull qwen:7b

# 第3步：运行推理
cd /mnt/nlp/yuanmengying/nli2hallucination/data/bert-classifier

# 使用50个样本快速测试
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --output_dir ./llm_results/llama3_test \
  --sample_size 50

# 使用全部900个测试样本
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --output_dir ./llm_results/llama3_full
```

### 运行多个模型进行对比

```bash
# 创建对比目录
mkdir -p comparison_results

# 测试llama3
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --output_dir ./comparison_results/llama3 \
  --sample_size 100

# 测试qwen
python llm_inference.py \
  --model_name qwen:7b \
  --deploy_type ollama \
  --output_dir ./comparison_results/qwen7b \
  --sample_size 100

# 测试qwen 14b
python llm_inference.py \
  --model_name qwen:14b \
  --deploy_type ollama \
  --output_dir ./comparison_results/qwen14b \
  --sample_size 100
```

---

## 详细用法

### 命令行参数

#### 模型配置参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_name` | 模型名称（必需） | `llama3`, `qwen:7b`, `qwen:14b` |
| `--deploy_type` | 部署方式 | `ollama`, `huggingface`, `api` |
| `--model_path` | HuggingFace模型路径 | `./models/llama3` |
| `--ollama_url` | Ollama服务地址 | `http://localhost:11434` |
| `--api_key` | API密钥 | `sk-xxx...` |
| `--api_url` | API地址 | `https://api.together.xyz` |

#### 数据配置参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--data_path` | 测试数据路径 | `../summary_nli_hallucination_dataset.xlsx` |
| `--sample_size` | 抽样大小 | `None`（使用全部） |
| `--output_dir` | 结果保存目录 | `./llm_results` |

#### 推理配置参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--use_zh_prompt` | 使用中文提示词 | `False` |
| `--temperature` | 采样温度（0=确定性） | `0.0` |
| `--max_tokens` | 最大生成token数 | `100` |

### 完整使用示例

#### 示例1：使用llama3进行推理（英文提示词）

```bash
python llm_inference.py \
  --model_name llama3 \
  --deploy_type ollama \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./results/llama3_en \
  --sample_size 200 \
  --temperature 0.0 \
  --max_tokens 50
```

#### 示例2：使用qwen进行推理（中文提示词）

```bash
python llm_inference.py \
  --model_name qwen:7b \
  --deploy_type ollama \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./results/qwen_zh \
  --use_zh_prompt \
  --sample_size 200
```

#### 示例3：使用HuggingFace模型进行推理

```bash
python llm_inference.py \
  --model_name llama3 \
  --deploy_type huggingface \
  --model_path /path/to/local/llama3 \
  --data_path ../summary_nli_hallucination_dataset.xlsx \
  --output_dir ./results/llama3_hf \
  --sample_size 100
```

#### 示例4：使用API进行推理（阿里云Qwen）

```bash
python llm_inference.py \
  --model_name qwen \
  --deploy_type api \
  --api_key "sk-xxx..." \
  --api_url https://dashscope.aliyuncs.com \
  --output_dir ./results/qwen_api \
  --sample_size 100
```

---

## 输出结果

### 输出文件说明

运行推理后，会在指定的 `--output_dir` 目录下生成以下文件：

#### 1. `llm_results.json` - 推理结果总结

```json
{
  "model_name": "llama3",
  "deploy_type": "ollama",
  "inference_time": "2024-01-15T10:30:00",
  "test_size": 900,
  "valid_predictions": 890,
  "use_zh_prompt": false,
  "detailed_metrics": {
    "accuracy": 0.7651,
    "macro_precision": 0.6234,
    "macro_recall": 0.6123,
    "macro_f1": 0.6178,
    "no_hallucination": {
      "precision": 0.8234,
      "recall": 0.8567,
      "f1_score": 0.8398,
      "support": 630
    },
    "hallucination": {
      "precision": 0.5145,
      "recall": 0.4234,
      "f1_score": 0.4646,
      "support": 270
    },
    "confusion_matrix": {
      "true_negatives": 540,
      "false_positives": 90,
      "false_negatives": 156,
      "true_positives": 104
    },
    "specificity": 0.857,
    "sensitivity": 0.4
  }
}
```

#### 2. `llm_detailed_predictions.xlsx` - 详细预测结果

包含以下列：
- `id`: 样本ID
- `context`: 上下文
- `output`: 生成的文本
- `label`: 真实标签（0=无幻觉，1=有幻觉）
- `llm_prediction`: LLM预测标签
- `llm_confidence`: LLM预测置信度
- `llm_raw_output`: LLM原始输出
- `correct_prediction`: 预测是否正确

### 结果解读

推理完成后，会在终端输出以下信息：

```
======================================================================
LLAMA3 幻觉检测推理结果
======================================================================

📊 总体性能指标:
准确率 (Accuracy): 0.7651
宏平均精确率: 0.6234
宏平均召回率: 0.6123
宏平均F1分数: 0.6178

🔍 无幻觉类别 (标签0):
精确率: 0.8234, 召回率: 0.8567, F1: 0.8398

⚠️  有幻觉类别 (标签1):
精确率: 0.5145, 召回率: 0.4234, F1: 0.4646

📈 关键指标:
敏感性 (Sensitivity): 0.4000
特异性 (Specificity): 0.8571

🔢 混淆矩阵:
真阴性 (TN): 540, 假阳性 (FP): 90
假阴性 (FN): 156, 真阳性 (TP): 104

💾 结果已保存到: ./llm_results/llama3
======================================================================
```

---

## 对比分析

### 对比BERT和LLM结果

```python
import pandas as pd
import json

# 加载BERT结果
bert_results = pd.read_excel('./test_results/detailed_predictions.xlsx')

# 加载LLM结果
llm_results = pd.read_excel('./llm_results/llama3/llm_detailed_predictions.xlsx')

# 合并结果
comparison = bert_results.merge(
    llm_results[['id', 'llm_prediction', 'llm_confidence']],
    on='id',
    how='inner'
)

# 计算一致性
agreement = (comparison['predicted_label'] == comparison['llm_prediction']).mean()
print(f"BERT和LLM预测一致率: {agreement:.2%}")

# 统计分歧情况
disagreement = comparison[comparison['predicted_label'] != comparison['llm_prediction']]
print(f"总分歧数: {len(disagreement)}")

# 保存对比结果
comparison.to_excel('./comparison_bert_llm.xlsx', index=False)
```

### Python脚本对比多个模型

```python
import json
import pandas as pd
from pathlib import Path

def compare_models(results_dirs):
    """对比多个模型的结果"""

    results = {}

    for model_dir in results_dirs:
        model_name = Path(model_dir).name

        # 加载结果
        with open(f'{model_dir}/llm_results.json', 'r', encoding='utf-8') as f:
            results[model_name] = json.load(f)

    # 创建对比表格
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': result['detailed_metrics']['accuracy'],
            'Precision': result['detailed_metrics']['hallucination']['precision'],
            'Recall': result['detailed_metrics']['hallucination']['recall'],
            'F1': result['detailed_metrics']['hallucination']['f1_score'],
        }
        for model, result in results.items()
    }).T

    print(comparison_df.to_string())
    return comparison_df

# 使用示例
model_dirs = [
    './comparison_results/llama3',
    './comparison_results/qwen7b',
    './comparison_results/qwen14b',
]

comparison = compare_models(model_dirs)
```

---

## 常见问题

### Q1: Ollama 连接超时

**问题**：`无法连接到Ollama服务 http://localhost:11434`

**解决**：
```bash
# 检查Ollama是否已启动
ollama serve

# 检查服务状态
curl http://localhost:11434/api/tags

# 如果无法启动，尝试重新安装
ollama --version
```

### Q2: 显存不足

**问题**：`CUDA out of memory`

**解决方案**：
1. 使用Ollama（自动优化内存）
2. 减小 `--sample_size`
3. 使用更小的模型（如 `qwen:7b` 而非 `qwen:14b`）
4. 使用API方式（不占用本地显存）

### Q3: 模型推理速度慢

**原因和优化方案**：

| 原因 | 优化方案 |
|------|--------|
| 模型太大 | 使用更小的模型或减少采样大小 |
| 硬件配置低 | 使用更快的硬件或API服务 |
| 网络延迟（API） | 选择更近的服务器或本地部署 |

### Q4: LLM预测准确率偏低

**可能原因**：
- Prompt设计不佳 - 修改 `HALLUCINATION_PROMPT_TEMPLATE` 或 `HALLUCINATION_PROMPT_ZH`
- 模型选择不当 - 尝试更大的模型
- 任务复杂度 - 数据中的幻觉类型复杂，LLM可能较难判断

**优化建议**：
```python
# 修改prompt以获得更好的结果
CUSTOM_PROMPT = """你是一位专业的文本质量评估专家。
请判断以下生成文本中是否存在与上下文不符的内容：

上下文：{context}

生成文本：{output}

请从以下几个方面检查：
1. 是否有事实错误
2. 是否有信息遗漏或添加
3. 是否有逻辑矛盾

答案："""
```

---

## 性能参考

### 不同部署方式的性能对比

| 部署方式 | 推理速度 | 显存占用 | 易用性 |
|---------|--------|--------|------|
| Ollama | 中等 | 4-8GB | 最简单 |
| HuggingFace | 快 | 8-16GB | 中等 |
| API | 慢 | 0 | 最简单 |

### 不同模型的性能参考

| 模型 | 参数 | 推理速度 | 质量 | 显存 |
|------|------|--------|------|------|
| llama3 | 8B | 中等 | 良好 | 6-8GB |
| llama3 | 70B | 慢 | 优秀 | 40GB+ |
| qwen | 7B | 快 | 良好 | 6GB |
| qwen | 14B | 中等 | 优秀 | 12GB |
| qwen | 72B | 慢 | 优秀 | 40GB+ |

---

## 扩展和自定义

### 添加新的LLM模型

```python
# 在 llm_inference.py 中修改 LLMInference 类

def predict_custom_llm(self, context: str, output: str, use_zh: bool = False) -> Dict:
    """自定义LLM推理"""
    prompt = self._get_prompt(context, output, use_zh)

    # 调用你的自定义模型
    # ...

    return {
        'prediction': prediction,
        'confidence': confidence,
        'raw_output': generated_text
    }
```

### 修改Prompt模板

```python
# 在 LLMInference 类中修改这两个常量

HALLUCINATION_PROMPT_TEMPLATE = """Your custom English prompt..."""

HALLUCINATION_PROMPT_ZH = """你的自定义中文提示词..."""
```

### 添加自定义评估指标

```python
def calculate_custom_metrics(y_true, y_pred):
    """添加自定义评估指标"""
    # 实现你的评估逻辑
    return custom_metrics
```

---

## 参考链接

- [Ollama官网](https://ollama.ai)
- [Meta Llama文档](https://github.com/facebookresearch/llama)
- [Qwen官方仓库](https://github.com/QwenLM/Qwen)
- [HuggingFace模型库](https://huggingface.co)

---

## 反馈和改进

如有问题或建议，欢迎提出改进意见！
