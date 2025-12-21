#!/usr/bin/env python3
"""
数据处理脚本：分析不同task_type的数据分布
输出格式：JSON, CSV, XLSX
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
import os

def load_data():
    """加载训练和测试数据"""
    train_path = "train-00000-of-00001.parquet"
    test_path = "test-00000-of-00001.parquet"
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # 添加数据集标识
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df, train_df, test_df

def analyze_task_type_distribution(df):
    """分析不同task_type的详细分布"""
    results = {}
    
    # 总体统计
    task_types = df['task_type'].unique()
    
    for task_type in task_types:
        task_data = df[df['task_type'] == task_type]
        
        # 基本统计
        basic_stats = {
            'total_samples': len(task_data),
            'percentage': len(task_data) / len(df) * 100,
            'train_samples': len(task_data[task_data['dataset'] == 'train']),
            'test_samples': len(task_data[task_data['dataset'] == 'test'])
        }
        
        # 质量分布
        quality_dist = task_data['quality'].value_counts().to_dict()
        quality_percentage = (task_data['quality'].value_counts() / len(task_data) * 100).to_dict()
        
        # 模型分布
        model_dist = task_data['model'].value_counts().to_dict()
        model_percentage = (task_data['model'].value_counts() / len(task_data) * 100).to_dict()
        
        # 幻觉统计
        hallucination_stats = analyze_hallucinations(task_data)
        
        # 文本长度统计
        text_length_stats = {
            'query_length': {
                'mean': float(task_data['query'].str.len().mean()),
                'median': float(task_data['query'].str.len().median()),
                'min': int(task_data['query'].str.len().min()),
                'max': int(task_data['query'].str.len().max()),
                'std': float(task_data['query'].str.len().std())
            },
            'context_length': {
                'mean': float(task_data['context'].str.len().mean()),
                'median': float(task_data['context'].str.len().median()),
                'min': int(task_data['context'].str.len().min()),
                'max': int(task_data['context'].str.len().max()),
                'std': float(task_data['context'].str.len().std())
            },
            'output_length': {
                'mean': float(task_data['output'].str.len().mean()),
                'median': float(task_data['output'].str.len().median()),
                'min': int(task_data['output'].str.len().min()),
                'max': int(task_data['output'].str.len().max()),
                'std': float(task_data['output'].str.len().std())
            }
        }
        
        # 温度参数统计
        temp_stats = {
            'mean': float(task_data['temperature'].mean()),
            'median': float(task_data['temperature'].median()),
            'min': float(task_data['temperature'].min()),
            'max': float(task_data['temperature'].max()),
            'std': float(task_data['temperature'].std())
        }
        
        results[task_type] = {
            'basic_statistics': basic_stats,
            'quality_distribution': {
                'counts': quality_dist,
                'percentages': {k: round(v, 2) for k, v in quality_percentage.items()}
            },
            'model_distribution': {
                'counts': model_dist,
                'percentages': {k: round(v, 2) for k, v in model_percentage.items()}
            },
            'hallucination_statistics': hallucination_stats,
            'text_length_statistics': text_length_stats,
            'temperature_statistics': temp_stats
        }
    
    return results

def analyze_hallucinations(task_data):
    """分析幻觉标签统计"""
    evident_conflict_counts = []
    baseless_info_counts = []
    
    for _, row in task_data.iterrows():
        labels = row['hallucination_labels_processed']
        if isinstance(labels, dict):
            evident_conflict_counts.append(labels.get('evident_conflict', 0))
            baseless_info_counts.append(labels.get('baseless_info', 0))
        else:
            evident_conflict_counts.append(0)
            baseless_info_counts.append(0)
    
    evident_conflict_counts = np.array(evident_conflict_counts)
    baseless_info_counts = np.array(baseless_info_counts)
    
    return {
        'evident_conflict': {
            'samples_with_conflict': int(np.sum(evident_conflict_counts > 0)),
            'percentage_with_conflict': float(np.sum(evident_conflict_counts > 0) / len(task_data) * 100),
            'total_conflicts': int(evident_conflict_counts.sum()),
            'mean_per_sample': float(evident_conflict_counts.mean())
        },
        'baseless_info': {
            'samples_with_baseless': int(np.sum(baseless_info_counts > 0)),
            'percentage_with_baseless': float(np.sum(baseless_info_counts > 0) / len(task_data) * 100),
            'total_baseless': int(baseless_info_counts.sum()),
            'mean_per_sample': float(baseless_info_counts.mean())
        },
        'any_hallucination': {
            'samples_with_any': int(np.sum((evident_conflict_counts > 0) | (baseless_info_counts > 0))),
            'percentage_with_any': float(np.sum((evident_conflict_counts > 0) | (baseless_info_counts > 0)) / len(task_data) * 100)
        }
    }

def create_summary_table(results):
    """创建汇总表格用于CSV和XLSX导出"""
    summary_data = []
    
    for task_type, stats in results.items():
        row = {
            'Task_Type': task_type,
            'Total_Samples': stats['basic_statistics']['total_samples'],
            'Percentage': round(stats['basic_statistics']['percentage'], 2),
            'Train_Samples': stats['basic_statistics']['train_samples'],
            'Test_Samples': stats['basic_statistics']['test_samples'],
            
            # 质量分布
            'Good_Quality': stats['quality_distribution']['counts'].get('good', 0),
            'Good_Quality_Pct': stats['quality_distribution']['percentages'].get('good', 0),
            
            # 幻觉统计
            'Evident_Conflict_Samples': stats['hallucination_statistics']['evident_conflict']['samples_with_conflict'],
            'Evident_Conflict_Pct': round(stats['hallucination_statistics']['evident_conflict']['percentage_with_conflict'], 2),
            'Baseless_Info_Samples': stats['hallucination_statistics']['baseless_info']['samples_with_baseless'],
            'Baseless_Info_Pct': round(stats['hallucination_statistics']['baseless_info']['percentage_with_baseless'], 2),
            'Any_Hallucination_Samples': stats['hallucination_statistics']['any_hallucination']['samples_with_any'],
            'Any_Hallucination_Pct': round(stats['hallucination_statistics']['any_hallucination']['percentage_with_any'], 2),
            
            # 文本长度统计
            'Avg_Query_Length': round(stats['text_length_statistics']['query_length']['mean'], 1),
            'Avg_Context_Length': round(stats['text_length_statistics']['context_length']['mean'], 1),
            'Avg_Output_Length': round(stats['text_length_statistics']['output_length']['mean'], 1),
            
            # 温度统计
            'Avg_Temperature': round(stats['temperature_statistics']['mean'], 3),
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def export_results(results, summary_df):
    """导出结果到多种格式"""
    # 1. 导出详细JSON
    with open('task_type_analysis_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 2. 导出汇总CSV
    summary_df.to_csv('task_type_analysis_summary.csv', index=False, encoding='utf-8')
    
    # 3. 导出汇总XLSX (需要安装openpyxl)
    try:
        with pd.ExcelWriter('task_type_analysis_summary.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Task_Type_Summary', index=False)
            
            # 为每个task_type创建详细sheet
            for task_type, stats in results.items():
                # 创建详细统计表
                detail_data = []
                
                # 基本统计
                detail_data.append(['基本统计', '', ''])
                detail_data.append(['总样本数', stats['basic_statistics']['total_samples'], ''])
                detail_data.append(['百分比', f"{stats['basic_statistics']['percentage']:.2f}%", ''])
                detail_data.append(['训练样本', stats['basic_statistics']['train_samples'], ''])
                detail_data.append(['测试样本', stats['basic_statistics']['test_samples'], ''])
                detail_data.append(['', '', ''])
                
                # 质量分布
                detail_data.append(['质量分布', '数量', '百分比'])
                for quality, count in stats['quality_distribution']['counts'].items():
                    pct = stats['quality_distribution']['percentages'][quality]
                    detail_data.append([quality, count, f"{pct:.2f}%"])
                detail_data.append(['', '', ''])
                
                # 模型分布
                detail_data.append(['模型分布', '数量', '百分比'])
                for model, count in stats['model_distribution']['counts'].items():
                    pct = stats['model_distribution']['percentages'][model]
                    detail_data.append([model, count, f"{pct:.2f}%"])
                
                detail_df = pd.DataFrame(detail_data, columns=['指标', '值', '百分比'])
                detail_df.to_excel(writer, sheet_name=f'{task_type}_详细', index=False)
        
        print("✅ 成功导出 XLSX 文件")
    except ImportError:
        print("⚠️  需要安装 openpyxl 才能导出 XLSX 文件: pip install openpyxl")

def main():
    """主函数"""
    print("🔄 正在加载数据...")
    df, train_df, test_df = load_data()
    
    print(f"📊 数据加载完成: 总样本 {len(df)}, 训练 {len(train_df)}, 测试 {len(test_df)}")
    
    print("🔄 正在分析不同task_type的分布...")
    results = analyze_task_type_distribution(df)
    
    print("🔄 正在创建汇总表格...")
    summary_df = create_summary_table(results)
    
    print("🔄 正在导出结果...")
    export_results(results, summary_df)
    
    print("\n✅ 分析完成！生成的文件:")
    print("   - task_type_analysis_detailed.json (详细JSON数据)")
    print("   - task_type_analysis_summary.csv (汇总CSV表格)")
    print("   - task_type_analysis_summary.xlsx (汇总Excel表格，包含详细sheets)")
    
    print("\n📈 快速汇总:")
    print(summary_df[['Task_Type', 'Total_Samples', 'Percentage', 'Any_Hallucination_Pct']].to_string(index=False))

if __name__ == "__main__":
    main()