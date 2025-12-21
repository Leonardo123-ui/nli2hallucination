#!/usr/bin/env python3
"""
创建Summary类别的NLI幻觉检测数据集
输出格式：xlsx
字段：id, context, output, label, split, task_type
"""

import pandas as pd
import json
import ast
from pathlib import Path

def load_summary_data():
    """加载Summary类别的数据"""
    # 从原始parquet文件加载并筛选Summary数据
    train_df = pd.read_parquet("train-00000-of-00001.parquet")
    test_df = pd.read_parquet("test-00000-of-00001.parquet")
    
    # 筛选Summary类别
    train_summary = train_df[train_df['task_type'] == 'Summary'].copy()
    test_summary = test_df[test_df['task_type'] == 'Summary'].copy()
    
    # 添加split标识
    train_summary['split'] = 'train'
    test_summary['split'] = 'test'
    
    # 合并
    summary_df = pd.concat([train_summary, test_summary], ignore_index=True)
    
    return summary_df

def process_hallucination_labels(hallucination_labels):
    """
    处理幻觉标签，如果任何标签中有1就认为是幻觉数据
    返回：1表示有幻觉，0表示无幻觉
    """
    try:
        # 处理字符串格式的字典
        if isinstance(hallucination_labels, str):
            labels_dict = ast.literal_eval(hallucination_labels)
        elif isinstance(hallucination_labels, dict):
            labels_dict = hallucination_labels
        else:
            return 0
        
        # 检查是否有任何标签值为1或大于0
        for key, value in labels_dict.items():
            if isinstance(value, (int, float)) and value > 0:
                return 1
        
        return 0
    except:
        # 如果解析失败，默认为0（无幻觉）
        return 0

def create_nli_dataset(summary_df):
    """创建NLI数据集"""
    nli_data = []
    
    for idx, row in summary_df.iterrows():
        # 创建唯一ID
        nli_id = f"summary_{row['split']}_{idx}"
        
        # 处理幻觉标签
        label = process_hallucination_labels(row['hallucination_labels_processed'])
        
        nli_record = {
            'id': nli_id,
            'context': row['context'],
            'output': row['output'],
            'label': label,
            'split': row['split'],
            'task_type': row['task_type']
        }
        
        nli_data.append(nli_record)
    
    return pd.DataFrame(nli_data)

def analyze_dataset_statistics(nli_df):
    """分析数据集统计信息"""
    stats = {
        'total_samples': len(nli_df),
        'hallucination_samples': len(nli_df[nli_df['label'] == 1]),
        'no_hallucination_samples': len(nli_df[nli_df['label'] == 0]),
        'hallucination_rate': len(nli_df[nli_df['label'] == 1]) / len(nli_df) * 100,
        'train_samples': len(nli_df[nli_df['split'] == 'train']),
        'test_samples': len(nli_df[nli_df['split'] == 'test']),
        'train_hallucination_rate': len(nli_df[(nli_df['split'] == 'train') & (nli_df['label'] == 1)]) / len(nli_df[nli_df['split'] == 'train']) * 100,
        'test_hallucination_rate': len(nli_df[(nli_df['split'] == 'test') & (nli_df['label'] == 1)]) / len(nli_df[nli_df['split'] == 'test']) * 100
    }
    return stats

def save_nli_dataset(nli_df, stats):
    """保存NLI数据集为xlsx格式"""
    filename = "summary_nli_hallucination_dataset.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 主数据集
        nli_df.to_excel(writer, sheet_name='NLI数据集', index=False)
        
        # 统计信息
        stats_data = {
            '统计指标': [
                '总样本数', 
                '幻觉样本数', 
                '非幻觉样本数', 
                '幻觉率(%)', 
                '训练集样本数', 
                '测试集样本数',
                '训练集幻觉率(%)',
                '测试集幻觉率(%)'
            ],
            '数值': [
                stats['total_samples'],
                stats['hallucination_samples'],
                stats['no_hallucination_samples'],
                f"{stats['hallucination_rate']:.2f}%",
                stats['train_samples'],
                stats['test_samples'],
                f"{stats['train_hallucination_rate']:.2f}%",
                f"{stats['test_hallucination_rate']:.2f}%"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='数据集统计', index=False)
        
        # 按split分组的数据
        train_data = nli_df[nli_df['split'] == 'train']
        test_data = nli_df[nli_df['split'] == 'test']
        
        train_data.to_excel(writer, sheet_name='训练集', index=False)
        test_data.to_excel(writer, sheet_name='测试集', index=False)
        
        # 标签分布分析
        label_stats = []
        for split in ['train', 'test']:
            split_data = nli_df[nli_df['split'] == split]
            label_counts = split_data['label'].value_counts().sort_index()
            
            for label, count in label_counts.items():
                percentage = count / len(split_data) * 100
                label_name = "有幻觉" if label == 1 else "无幻觉"
                label_stats.append({
                    '数据集': split,
                    '标签': label_name,
                    '标签值': label,
                    '数量': count,
                    '百分比': f"{percentage:.2f}%"
                })
        
        label_df = pd.DataFrame(label_stats)
        label_df.to_excel(writer, sheet_name='标签分布', index=False)
    
    return filename

def main():
    """主函数"""
    print("🔄 正在加载Summary类别数据...")
    summary_df = load_summary_data()
    
    print(f"📊 Summary数据加载完成: 总样本 {len(summary_df)}")
    print(f"   训练集: {len(summary_df[summary_df['split'] == 'train'])}")
    print(f"   测试集: {len(summary_df[summary_df['split'] == 'test'])}")
    
    print("🔄 正在创建NLI幻觉检测数据集...")
    nli_df = create_nli_dataset(summary_df)
    
    print("🔄 正在分析数据集统计信息...")
    stats = analyze_dataset_statistics(nli_df)
    
    print("🔄 正在保存数据集...")
    filename = save_nli_dataset(nli_df, stats)
    
    print("\n✅ NLI幻觉检测数据集创建完成！")
    print(f"📁 保存文件: {filename}")
    
    print(f"\n📊 数据集统计:")
    print(f"   总样本数: {stats['total_samples']}")
    print(f"   幻觉样本数: {stats['hallucination_samples']} ({stats['hallucination_rate']:.2f}%)")
    print(f"   非幻觉样本数: {stats['no_hallucination_samples']} ({100-stats['hallucination_rate']:.2f}%)")
    print(f"   训练集幻觉率: {stats['train_hallucination_rate']:.2f}%")
    print(f"   测试集幻觉率: {stats['test_hallucination_rate']:.2f}%")
    
    # 显示样本预览
    print(f"\n📋 数据集预览:")
    print("字段名称: id, context, output, label, split, task_type")
    print("标签说明: 0=无幻觉, 1=有幻觉")
    
    # 显示前几个样本的基本信息
    print(f"\n样本示例:")
    for i in range(min(3, len(nli_df))):
        row = nli_df.iloc[i]
        context_preview = row['context'][:100] + "..." if len(row['context']) > 100 else row['context']
        output_preview = row['output'][:100] + "..." if len(row['output']) > 100 else row['output']
        print(f"  ID: {row['id']}")
        print(f"  Context: {context_preview}")
        print(f"  Output: {output_preview}")
        print(f"  Label: {row['label']} ({'有幻觉' if row['label'] == 1 else '无幻觉'})")
        print(f"  Split: {row['split']}")
        print("  ---")

if __name__ == "__main__":
    main()