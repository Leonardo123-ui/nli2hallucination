#!/usr/bin/env python3
"""
幻觉检测器配置文件
"""

# 支持的模型列表
SUPPORTED_MODELS = {
    'bert-base-uncased': {
        'name': 'BERT Base Uncased',
        'max_length': 512,
        'description': '基础BERT模型，适合英文文本'
    },
    'bert-base-cased': {
        'name': 'BERT Base Cased', 
        'max_length': 512,
        'description': '区分大小写的BERT模型'
    },
    'roberta-base': {
        'name': 'RoBERTa Base',
        'max_length': 512,
        'description': 'RoBERTa基础模型，通常性能更好'
    },
    'roberta-large': {
        'name': 'RoBERTa Large',
        'max_length': 512,
        'description': 'RoBERTa大模型，性能最佳但计算量大'
    },
    'distilbert-base-uncased': {
        'name': 'DistilBERT Base',
        'max_length': 512,
        'description': '蒸馏版BERT，速度更快但精度略低'
    },
    'microsoft/deberta-base': {
        'name': 'DeBERTa Base',
        'max_length': 512,
        'description': 'DeBERTa模型，在多个任务上表现优秀'
    }
}

# 默认训练配置
DEFAULT_TRAIN_CONFIG = {
    'model_name': 'bert-base-uncased',
    'input_format': 'context_output',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'save_steps': 500,
    'eval_steps': 500,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'fp16': False,  # 混合精度训练，可节省显存
    'dataloader_num_workers': 4,
    'seed': 42
}

# 默认测试配置
DEFAULT_TEST_CONFIG = {
    'input_format': 'context_output',
    'max_length': 512,
    'batch_size': 32,
    'generate_plots': True,
    'save_predictions': True,
    'error_analysis': True
}

# 输入格式说明
INPUT_FORMATS = {
    'context_output': {
        'description': '使用上下文和输出: "[CLS] context [SEP] output [SEP]"',
        'template': '{context} [SEP] {output}',
        'recommended': True
    },
    'context_only': {
        'description': '仅使用上下文进行预测',
        'template': '{context}',
        'recommended': False
    },
    'output_only': {
        'description': '仅使用输出进行预测',
        'template': '{output}',
        'recommended': False
    }
}

# 评估指标说明
METRICS_DESCRIPTION = {
    'accuracy': '总体准确率：(TP+TN)/(TP+TN+FP+FN)',
    'balanced_accuracy': '平衡准确率：(Sensitivity+Specificity)/2',
    'precision': '精确率：TP/(TP+FP)，预测为正例中真正正例的比例',
    'recall': '召回率/敏感性：TP/(TP+FN)，真正正例中被预测为正例的比例',
    'f1_score': 'F1分数：2*Precision*Recall/(Precision+Recall)',
    'specificity': '特异性：TN/(TN+FP)，真正负例中被预测为负例的比例',
    'auc_score': 'AUC-ROC：ROC曲线下的面积，衡量分类器区分能力',
    'false_positive_rate': '假阳性率/误报率：FP/(FP+TN)',
    'false_negative_rate': '假阴性率/漏检率：FN/(FN+TP)',
}

# 幻觉检测特定指标
HALLUCINATION_METRICS = {
    'miss_rate': '幻觉漏检率：实际有幻觉但未被检出的比例',
    'false_alarm_rate': '幻觉误报率：实际无幻觉但被误判为有幻觉的比例',
    'hallucination_precision': '幻觉检测精确率：预测有幻觉中真正有幻觉的比例',
    'hallucination_recall': '幻觉检测召回率：真正有幻觉中被检测出的比例'
}

def get_model_info(model_name):
    """获取模型信息"""
    return SUPPORTED_MODELS.get(model_name, {
        'name': model_name,
        'max_length': 512,
        'description': '自定义模型'
    })

def validate_config(config, config_type='train'):
    """验证配置参数"""
    errors = []
    
    if config_type == 'train':
        required_fields = ['model_name', 'batch_size', 'learning_rate', 'num_epochs']
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
        
        if config.get('batch_size', 0) <= 0:
            errors.append("batch_size必须大于0")
        
        if config.get('learning_rate', 0) <= 0:
            errors.append("learning_rate必须大于0")
            
        if config.get('num_epochs', 0) <= 0:
            errors.append("num_epochs必须大于0")
    
    elif config_type == 'test':
        if config.get('batch_size', 0) <= 0:
            errors.append("batch_size必须大于0")
    
    # 检查输入格式
    if 'input_format' in config and config['input_format'] not in INPUT_FORMATS:
        errors.append(f"不支持的输入格式: {config['input_format']}")
    
    return errors

def print_config_info():
    """打印配置信息"""
    print("支持的模型:")
    for model_id, info in SUPPORTED_MODELS.items():
        print(f"  - {model_id}: {info['name']} - {info['description']}")
    
    print("\n输入格式选项:")
    for format_id, info in INPUT_FORMATS.items():
        recommended = " (推荐)" if info['recommended'] else ""
        print(f"  - {format_id}: {info['description']}{recommended}")
    
    print("\n评估指标说明:")
    for metric, desc in METRICS_DESCRIPTION.items():
        print(f"  - {metric}: {desc}")

if __name__ == "__main__":
    print_config_info()