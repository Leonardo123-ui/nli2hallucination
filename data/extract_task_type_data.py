#!/usr/bin/env python3
"""
提取不同task_type的具体数据并分别保存
输出格式：CSV（主要字段）和JSON（完整数据）
"""

import pandas as pd
import json
import os
from pathlib import Path

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
    return combined_df

def extract_task_type_data(df, task_type):
    """提取特定task_type的数据"""
    return df[df['task_type'] == task_type].copy()

def save_csv_format(task_data, task_type):
    """保存为CSV格式（主要字段）"""
    # 选择主要字段用于CSV导出
    csv_columns = [
        'dataset', 'task_type', 'model', 'temperature', 'quality',
        'query', 'context', 'output', 'hallucination_labels_processed'
    ]
    
    # 处理复杂字段
    csv_data = task_data[csv_columns].copy()
    
    # 将hallucination_labels_processed转换为字符串
    csv_data['hallucination_labels_processed'] = csv_data['hallucination_labels_processed'].astype(str)
    
    filename = f"{task_type.lower()}_data.csv"
    csv_data.to_csv(filename, index=False, encoding='utf-8')
    return filename

def save_json_format(task_data, task_type):
    """保存为JSON格式（完整数据）"""
    # 转换为字典列表
    records = task_data.to_dict('records')
    
    filename = f"{task_type.lower()}_data.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)
    return filename

def save_xlsx_format(task_data, task_type):
    """保存为Excel格式"""
    try:
        # 处理复杂字段
        xlsx_data = task_data.copy()
        xlsx_data['hallucination_labels_processed'] = xlsx_data['hallucination_labels_processed'].astype(str)
        
        filename = f"{task_type.lower()}_data.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 主数据sheet
            xlsx_data.to_excel(writer, sheet_name='完整数据', index=False)
            
            # 统计摘要sheet
            summary_data = {
                '指标': ['总样本数', '训练样本', '测试样本', '好质量样本', '好质量百分比'],
                '值': [
                    len(task_data),
                    len(task_data[task_data['dataset'] == 'train']),
                    len(task_data[task_data['dataset'] == 'test']),
                    len(task_data[task_data['quality'] == 'good']),
                    f"{len(task_data[task_data['quality'] == 'good']) / len(task_data) * 100:.2f}%"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
            
        return filename
    except ImportError:
        print(f"⚠️  需要安装 openpyxl 才能导出 {task_type} 的 XLSX 文件")
        return None

def create_summary_report(task_counts):
    """创建提取摘要报告"""
    report = {
        'extraction_summary': {
            'total_task_types': len(task_counts),
            'task_type_counts': task_counts,
            'files_generated': []
        }
    }
    
    with open('extraction_summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    print("🔄 正在加载数据...")
    df = load_data()
    
    print(f"📊 数据加载完成: 总样本 {len(df)}")
    
    # 获取所有task_type
    task_types = df['task_type'].unique()
    print(f"📋 发现的task_type: {list(task_types)}")
    
    task_counts = {}
    generated_files = []
    
    # 为每个task_type提取并保存数据
    for task_type in task_types:
        print(f"\n🔄 正在处理 {task_type}...")
        
        # 提取数据
        task_data = extract_task_type_data(df, task_type)
        task_counts[task_type] = len(task_data)
        
        print(f"   样本数量: {len(task_data)}")
        print(f"   训练集: {len(task_data[task_data['dataset'] == 'train'])}")
        print(f"   测试集: {len(task_data[task_data['dataset'] == 'test'])}")
        
        # 保存为多种格式
        csv_file = save_csv_format(task_data, task_type)
        json_file = save_json_format(task_data, task_type)
        xlsx_file = save_xlsx_format(task_data, task_type)
        
        generated_files.extend([csv_file, json_file])
        if xlsx_file:
            generated_files.append(xlsx_file)
        
        print(f"   ✅ 已保存: {csv_file}, {json_file}" + (f", {xlsx_file}" if xlsx_file else ""))
    
    # 创建摘要报告
    create_summary_report(task_counts)
    
    print(f"\n✅ 数据提取完成！")
    print(f"\n📁 生成的文件:")
    for file in generated_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   - {file} ({size_mb:.2f} MB)")
    
    print(f"\n📊 各task_type样本统计:")
    for task_type, count in task_counts.items():
        percentage = count / len(df) * 100
        print(f"   - {task_type}: {count} 样本 ({percentage:.2f}%)")

if __name__ == "__main__":
    main()