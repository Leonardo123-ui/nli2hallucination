import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, classification_report
)
import json
import argparse
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HallucinationDataset(Dataset):
    """幻觉检测数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_test_data(data_path):
    """加载测试数据"""
    logger.info(f"正在加载测试数据: {data_path}")
    
    # 加载完整数据集
    full_df = pd.read_excel(data_path, sheet_name='NLI数据集')
    train_df = full_df[full_df['split'] == 'train']
    test_df = full_df[full_df['split'] == 'test']
    
    logger.info(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    
    return train_df, test_df

def prepare_texts(df, input_format='context_output'):
    """准备输入文本"""
    if input_format == 'context_output':
        texts = [f"{row['context']} [SEP] {row['output']}" for _, row in df.iterrows()]
    elif input_format == 'context_only':
        texts = df['context'].tolist()
    elif input_format == 'output_only':
        texts = df['output'].tolist()
    else:
        raise ValueError(f"不支持的输入格式: {input_format}")
    
    return texts

def predict_batch(model, dataloader, device):
    """批量预测"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 获取概率
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_detailed_metrics(y_true, y_pred, y_prob):
    """计算详细的评估指标"""
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 各类别指标
    no_hall_precision, hall_precision = precision
    no_hall_recall, hall_recall = recall
    no_hall_f1, hall_f1 = f1
    no_hall_support, hall_support = support
    
    # 特异性和敏感性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 假阳性率和假阴性率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # AUC分数
    try:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc_score = 0.0
    
    # 平衡准确率
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return {
        # 总体指标
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'auc_score': auc_score,
        
        # 类别0（无幻觉）指标
        'no_hallucination': {
            'precision': no_hall_precision,
            'recall': no_hall_recall,
            'f1_score': no_hall_f1,
            'support': int(no_hall_support)
        },
        
        # 类别1（有幻觉）指标  
        'hallucination': {
            'precision': hall_precision,
            'recall': hall_recall,
            'f1_score': hall_f1,
            'support': int(hall_support)
        },
        
        # 混淆矩阵相关
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        
        # 率指标
        'specificity': specificity,
        'sensitivity': sensitivity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'miss_rate': fnr,  # 漏检率
        'fall_out': fpr    # 误报率
    }

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Hallucination', 'Hallucination'],
                yticklabels=['No Hallucination', 'Hallucination'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_score = roc_auc_score(y_true, y_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, output_dir):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_errors(df, predictions, labels, output_dir):
    """分析错误样本"""
    df_analysis = df.copy()
    df_analysis['predicted'] = predictions
    df_analysis['correct'] = (predictions == labels)
    
    # 错误样本
    errors = df_analysis[~df_analysis['correct']]
    
    # 假阳性（误报）：实际无幻觉，预测有幻觉
    false_positives = errors[errors['label'] == 0]
    
    # 假阴性（漏报）：实际有幻觉，预测无幻觉  
    false_negatives = errors[errors['label'] == 1]
    
    # 保存错误分析
    error_analysis = {
        'total_errors': len(errors),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'error_rate': len(errors) / len(df_analysis)
    }
    
    # 保存错误样本（前10个）
    if len(false_positives) > 0:
        fp_samples = false_positives.head(10)[['id', 'context', 'output', 'label', 'predicted']].to_dict('records')
        error_analysis['false_positive_samples'] = fp_samples
    
    if len(false_negatives) > 0:
        fn_samples = false_negatives.head(10)[['id', 'context', 'output', 'label', 'predicted']].to_dict('records')
        error_analysis['false_negative_samples'] = fn_samples
    
    with open(f'{output_dir}/error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    return error_analysis

def test_model(
    model_path,
    data_path='../summary_nli_hallucination_dataset.xlsx',
    output_dir='./test_results',
    input_format='context_output',
    max_length=512,
    batch_size=32
):
    """测试模型"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    logger.info(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # 加载数据
    train_df, test_df = load_test_data(data_path)
    
    # 准备测试数据
    test_texts = prepare_texts(test_df, input_format)
    test_labels = test_df['label'].tolist()
    
    # 创建数据集和数据加载器
    test_dataset = HallucinationDataset(test_texts, test_labels, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 进行预测
    logger.info("正在进行预测...")
    predictions, probabilities, true_labels = predict_batch(model, test_dataloader, device)
    
    # 计算指标
    logger.info("正在计算评估指标...")
    metrics = calculate_detailed_metrics(true_labels, predictions, probabilities)
    
    # 生成分类报告
    class_report = classification_report(
        true_labels, predictions, 
        target_names=['No Hallucination', 'Hallucination'],
        output_dict=True
    )
    
    # 绘制图表
    logger.info("正在生成可视化图表...")
    plot_confusion_matrix(true_labels, predictions, output_dir)
    plot_roc_curve(true_labels, probabilities, output_dir)
    plot_precision_recall_curve(true_labels, probabilities, output_dir)
    
    # 错误分析
    logger.info("正在进行错误分析...")
    error_analysis = analyze_errors(test_df, predictions, true_labels, output_dir)
    
    # 保存完整结果
    results = {
        'model_path': model_path,
        'test_time': datetime.now().isoformat(),
        'test_size': len(test_dataset),
        'input_format': input_format,
        'detailed_metrics': metrics,
        'classification_report': class_report,
        'error_analysis_summary': error_analysis
    }
    
    with open(f'{output_dir}/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存预测结果
    results_df = test_df.copy()
    results_df['predicted_label'] = predictions
    results_df['hallucination_probability'] = probabilities[:, 1]
    results_df['correct_prediction'] = (predictions == true_labels)
    
    results_df.to_excel(f'{output_dir}/detailed_predictions.xlsx', index=False)
    
    logger.info("测试完成！")
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description='测试幻觉检测模型')
    parser.add_argument('--model_path', required=True,
                       help='训练好的模型路径')
    parser.add_argument('--data_path', default='../summary_nli_hallucination_dataset.xlsx',
                       help='测试数据路径')
    parser.add_argument('--output_dir', default='./test_results',
                       help='结果保存目录')
    parser.add_argument('--input_format', default='context_output',
                       choices=['context_output', 'context_only', 'output_only'],
                       help='输入文本格式')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 执行测试
    metrics, results = test_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        input_format=args.input_format,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("幻觉检测模型测试结果")
    print("="*60)
    
    print(f"\n📊 总体性能指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}")
    print(f"宏平均精确率: {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
    print(f"宏平均F1分数: {metrics['macro_f1']:.4f}")
    print(f"AUC分数: {metrics['auc_score']:.4f}")
    
    print(f"\n🔍 无幻觉类别 (标签0):")
    no_hall = metrics['no_hallucination']
    print(f"精确率: {no_hall['precision']:.4f}")
    print(f"召回率: {no_hall['recall']:.4f}")
    print(f"F1分数: {no_hall['f1_score']:.4f}")
    print(f"样本数: {no_hall['support']}")
    
    print(f"\n⚠️  有幻觉类别 (标签1):")
    hall = metrics['hallucination']
    print(f"精确率: {hall['precision']:.4f}")
    print(f"召回率: {hall['recall']:.4f}")
    print(f"F1分数: {hall['f1_score']:.4f}")
    print(f"样本数: {hall['support']}")
    
    print(f"\n📈 关键幻觉检测指标:")
    print(f"敏感性/召回率 (Sensitivity): {metrics['sensitivity']:.4f}")
    print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
    print(f"假阳性率/误报率 (FPR): {metrics['false_positive_rate']:.4f}")
    print(f"假阴性率/漏检率 (FNR): {metrics['false_negative_rate']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\n🔢 混淆矩阵:")
    print(f"真阴性 (TN): {cm['true_negatives']}")
    print(f"假阳性 (FP): {cm['false_positives']}")
    print(f"假阴性 (FN): {cm['false_negatives']}")
    print(f"真阳性 (TP): {cm['true_positives']}")
    
    print(f"\n💾 结果已保存到: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()