#!/usr/bin/env python3
"""
幻觉检测器使用示例脚本
演示完整的训练和测试流程
"""

import os
import subprocess
import sys
from config import DEFAULT_TRAIN_CONFIG, DEFAULT_TEST_CONFIG, print_config_info

def run_training_example():
    """运行训练示例"""
    print("="*60)
    print("开始训练示例 - BERT幻觉检测器")
    print("="*60)
    
    # 检查数据文件是否存在
    data_file = "../summary_nli_hallucination_dataset.xlsx"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行 create_summary_nli_dataset.py 生成数据集")
        return False
    
    # 训练命令
    train_cmd = [
        sys.executable, "train_hallucination_detector.py",
        "--model_name", "bert-base-uncased",
        "--data_path", data_file,
        "--output_dir", "./models/bert_hallucination_detector",
        "--input_format", "context_output",
        "--batch_size", "8",  # 使用较小的batch_size以适应更多设备
        "--learning_rate", "2e-5",
        "--num_epochs", "2",  # 减少epoch数量用于快速测试
        "--save_steps", "250",
        "--eval_steps", "250"
    ]
    
    print("训练命令:", " ".join(train_cmd))
    print("开始训练...")
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print("✅ 训练成功完成")
        print("训练输出:", result.stdout[-500:])  # 打印最后500字符
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 训练失败")
        print("错误输出:", e.stderr)
        return False

def run_testing_example():
    """运行测试示例"""
    print("="*60)
    print("开始测试示例")
    print("="*60)
    
    model_path = "./models/bert_hallucination_detector/final_model"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本")
        return False
    
    data_file = "../summary_nli_hallucination_dataset.xlsx"
    
    # 测试命令
    test_cmd = [
        sys.executable, "test_hallucination_detector.py",
        "--model_path", model_path,
        "--data_path", data_file,
        "--output_dir", "./test_results/bert_results",
        "--input_format", "context_output",
        "--batch_size", "16"
    ]
    
    print("测试命令:", " ".join(test_cmd))
    print("开始测试...")
    
    try:
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        print("✅ 测试成功完成")
        print("测试输出:", result.stdout[-1000:])  # 打印最后1000字符
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 测试失败")
        print("错误输出:", e.stderr)
        return False

def run_quick_test():
    """运行快速测试 - 使用DistilBERT进行快速验证"""
    print("="*60)
    print("快速测试 - DistilBERT (适合资源有限的环境)")
    print("="*60)
    
    data_file = "../summary_nli_hallucination_dataset.xlsx"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 快速训练命令
    train_cmd = [
        sys.executable, "train_hallucination_detector.py",
        "--model_name", "distilbert-base-uncased",
        "--data_path", data_file,
        "--output_dir", "./models/distilbert_quick_test",
        "--input_format", "context_output",
        "--batch_size", "4",
        "--learning_rate", "3e-5",
        "--num_epochs", "1",
        "--save_steps", "100",
        "--eval_steps", "100"
    ]
    
    print("快速训练命令:", " ".join(train_cmd))
    
    try:
        print("开始快速训练...")
        subprocess.run(train_cmd, check=True)
        
        # 快速测试
        test_cmd = [
            sys.executable, "test_hallucination_detector.py",
            "--model_path", "./models/distilbert_quick_test/final_model",
            "--data_path", data_file,
            "--output_dir", "./test_results/distilbert_quick_test",
            "--batch_size", "8"
        ]
        
        print("开始快速测试...")
        subprocess.run(test_cmd, check=True)
        
        print("✅ 快速测试完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ 快速测试失败")
        print("错误:", e)
        return False

def check_requirements():
    """检查依赖包"""
    print("检查依赖包...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'scikit-learn',
        'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包都已安装")
    return True

def main():
    """主函数"""
    print("幻觉检测器示例脚本")
    print_config_info()
    
    print("\n" + "="*60)
    print("可用的示例:")
    print("1. 完整示例 (BERT训练+测试)")
    print("2. 快速测试 (DistilBERT)")
    print("3. 仅训练")
    print("4. 仅测试 (需要已有模型)")
    print("5. 检查依赖")
    print("="*60)
    
    while True:
        choice = input("\n请选择要运行的示例 (1-5, q退出): ").strip()
        
        if choice == 'q':
            break
        elif choice == '1':
            if check_requirements():
                if run_training_example():
                    run_testing_example()
        elif choice == '2':
            if check_requirements():
                run_quick_test()
        elif choice == '3':
            if check_requirements():
                run_training_example()
        elif choice == '4':
            if check_requirements():
                run_testing_example()
        elif choice == '5':
            check_requirements()
        else:
            print("无效选择，请输入1-5或q")

if __name__ == "__main__":
    main()