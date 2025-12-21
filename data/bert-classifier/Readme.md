# 幻觉检测器 (Hallucination Detector)

基于BERT/RoBERTa等预训练模型的文本幻觉检测系统，专门用于检测Summary任务中的幻觉现象。

## 项目结构

```
classifier/
├── train_hallucination_detector.py    # 训练脚本
├── test_hallucination_detector.py     # 测试评估脚本
├── config.py                          # 配置文件
├── run_example.py                     # 使用示例脚本
├── requirements.txt                   # 依赖包
└── README.md                          # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
python run_example.py
```

选择合适的示例模式：
- **选项1**: 完整示例 (BERT训练+测试)
- **选项2**: 快速测试 (DistilBERT，适合资源有限环境)

### 3. 手动训练

```bash
# 使用BERT训练
python train_hallucination_detector.py \
    --model_name ../../models/bert-base-uncased \
    --data_path ../summary_nli_hallucination_dataset.xlsx \
    --output_dir ./models/bert_hallucination_detector \
    --batch_size 16 \
    --num_epochs 3

# 使用RoBERTa训练（推荐）
python train_hallucination_detector.py \
    --model_name roberta-base \
    --data_path ../summary_nli_hallucination_dataset.xlsx \
    --output_dir ./models/roberta_hallucination_detector \
    --batch_size 16 \
    --num_epochs 3
```

### 4. 测试评估

```bash
python test_hallucination_detector.py \
    --model_path ./models/bert_hallucination_detector/final_model \
    --data_path ../summary_nli_hallucination_dataset.xlsx \
    --output_dir ./test_results/bert_results
```

## 支持的模型

| 模型 | 描述 | 推荐场景 |
|------|------|----------|
| `bert-base-uncased` | 基础BERT模型 | 通用场景，平衡性能和速度 |
| `roberta-base` | RoBERTa基础模型 | 推荐使用，通常性能更好 |
| `roberta-large` | RoBERTa大模型 | 最佳性能，需要更多计算资源 |
| `distilbert-base-uncased` | 蒸馏版BERT | 快速验证，资源有限环境 |
| `microsoft/deberta-base` | DeBERTa模型 | 在多个NLU任务上表现优秀 |

## 输入格式

### context_output (推荐)
使用上下文和输出的组合：`[CLS] context [SEP] output [SEP]`

### context_only  
仅使用上下文进行幻觉预测

### output_only
仅使用输出文本进行幻觉检测

## 评估指标

### 基础指标
- **Accuracy**: 总体准确率
- **Precision**: 精确率 (幻觉检测)
- **Recall**: 召回率 (幻觉检测)
- **F1-Score**: F1分数
- **AUC-ROC**: ROC曲线下面积

### 幻觉检测特定指标
- **Sensitivity**: 敏感性，检测出幻觉的能力
- **Specificity**: 特异性，正确识别无幻觉的能力
- **False Negative Rate**: 假阴性率，幻觉漏检率
- **False Positive Rate**: 假阳性率，幻觉误报率

### 混淆矩阵
```
                实际
预测    无幻觉  有幻觉
无幻觉    TN     FN
有幻觉    FP     TP
```

## 训练参数

### 关键参数
- `--model_name`: 预训练模型名称
- `--batch_size`: 批次大小 (建议: 8-32)
- `--learning_rate`: 学习率 (建议: 1e-5 到 5e-5)
- `--num_epochs`: 训练轮数 (建议: 2-5)
- `--max_length`: 最大序列长度 (默认: 512)

### 内存优化
- 减小 `batch_size`
- 使用 `--fp16` 启用混合精度训练
- 选择较小的模型如 `distilbert-base-uncased`

## 输出文件

### 训练输出
```
models/
└── model_name/
    ├── final_model/           # 最终模型
    ├── checkpoint-*/          # 检查点
    ├── training_info.json     # 训练信息
    └── logs/                  # 训练日志
```

### 测试输出
```
test_results/
├── test_results.json         # 完整评估结果
├── detailed_predictions.xlsx # 详细预测结果
├── error_analysis.json      # 错误分析
├── confusion_matrix.png     # 混淆矩阵图
├── roc_curve.png           # ROC曲线
└── precision_recall_curve.png # PR曲线
```

## 性能基准

基于Summary数据集 (5,658样本，29.8%幻觉率):

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|----|----|
| BERT-base | ~0.85 | ~0.78 | ~0.72 | ~0.75 | ~0.88 |
| RoBERTa-base | ~0.87 | ~0.80 | ~0.75 | ~0.77 | ~0.90 |
| DistilBERT | ~0.82 | ~0.74 | ~0.68 | ~0.71 | ~0.85 |

*注：实际性能可能因数据和参数而异*

## 使用建议

### 1. 模型选择
- **生产环境**: `roberta-base` 或 `roberta-large`
- **快速验证**: `distilbert-base-uncased`
- **资源充足**: `roberta-large` 或 `microsoft/deberta-base`

### 2. 超参数调优
- 学习率: 从 `2e-5` 开始，可尝试 `1e-5` 或 `3e-5`
- 批次大小: 根据GPU内存调整，通常8-32
- 训练轮数: 2-5轮，观察验证集性能

### 3. 数据增强
- 可以考虑使用更多的上下文信息
- 尝试不同的输入格式组合
- 使用更大的预训练模型

### 4. 错误分析
- 关注假阴性样本（漏检的幻觉）
- 分析假阳性样本（误报的情况）
- 根据错误模式调整模型或数据

## 故障排除

### 内存不足
```bash
# 减少批次大小
--batch_size 4

# 使用梯度累积
--gradient_accumulation_steps 4

# 启用混合精度
--fp16
```

### 训练速度慢
```bash
# 使用较小模型
--model_name distilbert-base-uncased

# 减少最大长度
--max_length 256

# 增加数据加载器工作线程
--dataloader_num_workers 4
```

### CUDA相关问题
```bash
# 检查CUDA版本
python -c "import torch; print(torch.version.cuda)"

# 检查GPU可用性
python -c "import torch; print(torch.cuda.is_available())"
```

## 扩展功能

### 自定义模型
可以使用任何Hugging Face上的序列分类模型：
```bash
python train_hallucination_detector.py \
    --model_name your-custom-model \
    --data_path your_data.xlsx
```

### 多GPU训练
```bash
# 使用torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_hallucination_detector.py \
    --your-args
```

### 与其他数据集结合
脚本支持任何包含 `context`, `output`, `label`, `split` 字段的Excel数据集。

## 引用

如果使用此代码，请引用相关论文和数据集：

```bibtex
@misc{hallucination_detector_2024,
  title={BERT-based Hallucination Detection for Text Summarization},
  author={Your Name},
  year={2024}
}