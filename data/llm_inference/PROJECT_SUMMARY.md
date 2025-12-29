# 📊 LLM 幻觉检测推理系统 - 完整项目说明

## ✨ 创建内容概览

为你的幻觉检测项目创建了一套完整的LLM推理系统，支持llama3和qwen3模型，包括3种部署方式和完整的文档。

### 📂 创建的文件清单

#### 🎯 核心推理脚本 (3个)

1. **llm_inference.py** (21KB) - 主推理脚本
   - ✅ 支持Ollama、HuggingFace、API三种部署方式
   - ✅ 支持llama3、qwen等多种LLM
   - ✅ 支持中英文提示词
   - ✅ 完整的评估指标计算
   - ✅ 预测结果导出

2. **config_examples.py** (9.1KB) - 配置示例和快速启动
   - ✅ 8个使用示例
   - ✅ 支持对比多个模型
   - ✅ 自动生成对比报告

3. **compare_models.py** (13KB) - 模型对比分析工具
   - ✅ BERT vs LLM对比
   - ✅ 一致性分析
   - ✅ 错误分析
   - ✅ 详细的对比报告

#### 📖 完整文档 (4个)

1. **QUICK_START.md** (4.3KB) - ⚡ 5分钟快速开始
   - 最简单的开始方式
   - 常用命令速查表
   - 常见问题解答

2. **LLM_INFERENCE_GUIDE.md** (15KB) - 📚 详细使用指南
   - 环境准备
   - 三种部署方式详解
   - 完整参数说明
   - 输出结果解读
   - 性能参考

3. **LLM_README.md** (6.2KB) - 📋 项目总览
   - 文件说明
   - 快速开始
   - 常见命令
   - 结果分析示例

4. **check_environment.py** (6.3KB) - 🔧 环境检查脚本
   - 检查Python版本
   - 检查依赖包
   - 检查CUDA/GPU
   - 检查Ollama服务
   - 检查数据文件

#### 📦 依赖配置

**requirements_llm.txt** (770B) - Python依赖列表

---

## 🚀 快速开始（3步）

### 第1步：安装Ollama（一次性）
```bash
curl https://ollama.ai/install.sh | sh
ollama serve  # 启动服务
```

### 第2步：拉取模型（一次性，另一个终端）
```bash
ollama pull llama3
ollama pull qwen:7b
```

### 第3步：运行推理
```bash
cd /mnt/nlp/yuanmengying/nli2hallucination/data/bert-classifier

# 快速测试（50个样本）
python llm_inference.py --model_name llama3 --sample_size 50

# 完整测试（900个样本）
python llm_inference.py --model_name llama3
```

**完成！** 结果保存在 `./llm_results/` 中

---

## 📋 使用场景和命令

### 场景1：快速验证
```bash
# 用50个样本快速测试
python llm_inference.py --model_name llama3 --sample_size 50
```

### 场景2：完整评估
```bash
# 使用全部900个测试样本
python llm_inference.py --model_name llama3
```

### 场景3：中文推理
```bash
# 使用中文提示词
python llm_inference.py --model_name qwen:7b --use_zh_prompt
```

### 场景4：模型对比
```bash
# 对比多个模型（需要分别运行）
python config_examples.py compare
```

### 场景5：BERT vs LLM对比
```bash
# 1. 先运行BERT推理（已有结果）
# 2. 再运行LLM推理
python llm_inference.py --model_name llama3

# 3. 对比分析
python compare_models.py
```

---

## 📊 输出结果说明

每次推理会生成：

```
./llm_results/
├── llm_results.json                   # 评估指标总结
│   ├── accuracy                       # 准确率
│   ├── detailed_metrics              # 详细指标
│   │   ├── hallucination.precision   # 幻觉精确率
│   │   ├── hallucination.recall      # 幻觉召回率
│   │   └── hallucination.f1_score    # 幻觉F1分数
│   └── confusion_matrix              # 混淆矩阵
│
└── llm_detailed_predictions.xlsx      # 详细预测结果
    ├── id                             # 样本ID
    ├── context                        # 上下文
    ├── output                         # 生成文本
    ├── label                          # 真实标签
    ├── llm_prediction                 # LLM预测
    ├── llm_confidence                 # LLM置信度
    └── correct_prediction             # 是否预测正确
```

### 示例输出

```
================================================================
LLAMA3 幻觉检测推理结果
================================================================

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
================================================================
```

---

## 🛠️ 推荐的工作流

### 第一周：验证和对比

```bash
# Day 1: 快速验证
python llm_inference.py --model_name llama3 --sample_size 50

# Day 2-3: 完整评估
python llm_inference.py --model_name llama3

# Day 4: 尝试qwen
python llm_inference.py --model_name qwen:7b --use_zh_prompt

# Day 5: 对比分析
python compare_models.py
```

### 第二周：深度分析

```bash
# 对比多个模型
python config_examples.py compare

# 分析错误分布
# 修改Prompt进行优化
# 尝试更大模型（14B, 70B）
```

### 后续：持续优化

```bash
# 根据对比结果选择最佳模型
# 调整Prompt提升效果
# 集成到生产流程
```

---

## 🔧 三种部署方式对比

| 方式 | 安装难度 | 硬件要求 | 推理速度 | 质量 | 推荐场景 |
|------|--------|--------|--------|------|--------|
| **Ollama** ⭐⭐⭐ | 最简单 | GPU 4GB+ | 中等 | 良好 | ✅ 快速开始 |
| **HuggingFace** ⭐⭐ | 中等 | GPU 8GB+ | 快 | 优秀 | 高级用户 |
| **API** ⭐⭐⭐⭐⭐ | 最简单 | 无 | 依赖网络 | 最优 | 无硬件限制 |

---

## 💡 关键特性

### ✨ 推理脚本特性
- ✅ **多模型支持**：llama3, qwen, chatglm等
- ✅ **多部署方式**：Ollama、HuggingFace、API
- ✅ **中英文支持**：内置中英文提示词
- ✅ **完整评估**：准确率、F1、混淆矩阵等
- ✅ **灵活配置**：温度、token等可调参数
- ✅ **结果导出**：JSON和Excel格式

### 📊 对比分析特性
- ✅ **一致性分析**：两个模型预测的一致率
- ✅ **错误分析**：分析分歧案例
- ✅ **详细报告**：生成对比报告和统计
- ✅ **Excel导出**：方便后续分析

### 📚 文档特性
- ✅ **快速开始**：5分钟上手
- ✅ **详细指南**：完整功能说明
- ✅ **代码示例**：8个使用示例
- ✅ **故障排查**：常见问题解答

---

## 🔍 支持的模型

### Ollama（推荐）
```bash
ollama pull llama3          # 8B，推荐首选
ollama pull qwen:7b         # 7B，适合中文
ollama pull qwen:14b        # 14B，更好的质量
ollama pull mistral         # Mistral，快速
```

### HuggingFace
- meta-llama/Llama-2-7b-chat-hf
- QwenLM/Qwen-7B-Chat
- THUDM/chatglm-6b
- meta-llama/Llama-2-70b-chat-hf（需要大显存）

### API
- 阿里云DashScope（Qwen）
- OpenAI / Together AI（Llama）

---

## 📈 性能参考

### 推理时间（单个样本）
| 模型 | Ollama | HuggingFace |
|------|--------|-----------|
| llama3 7B | ~5秒 | ~2秒 |
| qwen 7B | ~4秒 | ~2秒 |
| qwen 14B | ~8秒 | ~4秒 |
| llama3 70B | 30秒+ | 15秒+ |

### 显存占用
| 模型 | 显存 |
|------|------|
| llama3 7B | 4-6GB |
| qwen 7B | 4-6GB |
| qwen 14B | 10-12GB |
| llama3 70B | 40GB+ |

---

## 🚦 下一步建议

1. **立即开始**
   ```bash
   python check_environment.py  # 检查环境
   python llm_inference.py --model_name llama3 --sample_size 50  # 快速测试
   ```

2. **查看文档**
   - 快速开始：`QUICK_START.md`
   - 详细指南：`LLM_INFERENCE_GUIDE.md`
   - 项目说明：`LLM_README.md`

3. **尝试示例**
   ```bash
   python config_examples.py 1   # 运行示例1
   python config_examples.py 6   # 对比多个模型
   ```

4. **对比分析**
   ```bash
   python compare_models.py  # BERT vs LLM对比
   ```

---

## 📞 支持和反馈

### 遇到问题？
1. 查看 `QUICK_START.md` 的常见问题
2. 运行 `python check_environment.py` 检查环境
3. 查看命令帮助：`python llm_inference.py --help`

### 需要帮助？
- 快速开始：`QUICK_START.md`
- 详细指南：`LLM_INFERENCE_GUIDE.md`
- 示例代码：`config_examples.py`

---

## 📊 文件目录结构

```
bert-classifier/
├── 推理脚本 (核心)
│   ├── llm_inference.py              # 主推理脚本
│   ├── config_examples.py            # 配置示例
│   └── compare_models.py             # 对比工具
│
├── 文档 (使用指南)
│   ├── QUICK_START.md                # 快速开始
│   ├── LLM_INFERENCE_GUIDE.md        # 详细指南
│   ├── LLM_README.md                 # 项目总览
│   └── check_environment.py          # 环境检查
│
├── 依赖配置
│   └── requirements_llm.txt          # Python依赖
│
└── 现有文件 (BERT相关)
    ├── train.py
    ├── test.py
    ├── run.py
    └── models/                       # 已训练的BERT模型
```

---

## 🎉 祝你使用愉快！

现在你有了：
- ✅ 强大的LLM推理脚本
- ✅ 完整的文档和示例
- ✅ 灵活的部署选项
- ✅ 完善的对比分析工具

开始探索：`python llm_inference.py --help`

有问题？查看：`QUICK_START.md`

---

**创建日期**: 2024年
**系统**: LLM幻觉检测推理系统
**支持模型**: llama3, qwen系列
**支持部署**: Ollama, HuggingFace, API
