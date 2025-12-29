#!/usr/bin/env python3
"""
LLM推理配置示例
演示如何使用不同的模型配置进行推理
"""

from llm_inference import LLMConfig, ModelDeployType, run_inference
import os

def example_ollama_llama3():
    """示例1: 使用Ollama运行llama3"""
    print("="*60)
    print("示例1: Ollama + llama3")
    print("="*60)

    config = LLMConfig(
        model_name="llama3",
        deploy_type=ModelDeployType.OLLAMA,
        ollama_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/ollama_llama3",
        use_zh_prompt=False,
        sample_size=50  # 先用小样本测试
    )


def example_ollama_qwen():
    """示例2: 使用Ollama运行qwen"""
    print("="*60)
    print("示例2: Ollama + qwen:7b")
    print("="*60)

    config = LLMConfig(
        model_name="qwen:7b",
        deploy_type=ModelDeployType.OLLAMA,
        ollama_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/ollama_qwen7b",
        use_zh_prompt=True,  # 使用中文提示词
        sample_size=50
    )


def example_huggingface_llama3():
    """示例3: 使用HuggingFace加载llama3"""
    print("="*60)
    print("示例3: HuggingFace + llama3")
    print("="*60)

    config = LLMConfig(
        model_name="llama3",
        deploy_type=ModelDeployType.HUGGINGFACE,
        model_path="/mnt/nlp/models/Llama-2-7b-hf",  # 根据实际路径修改
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/hf_llama3",
        use_zh_prompt=False,
        sample_size=100
    )


def example_api_qwen():
    """示例4: 使用API调用阿里云Qwen"""
    print("="*60)
    print("示例4: 阿里云API + Qwen")
    print("="*60)

    # 确保设置了API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_name="qwen",
        deploy_type=ModelDeployType.API,
        api_key=api_key,
        api_url="https://dashscope.aliyuncs.com",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/api_qwen",
        use_zh_prompt=True,
        sample_size=50
    )


def example_api_openai_llama():
    """示例5: 使用OpenAI兼容API运行llama"""
    print("="*60)
    print("示例5: OpenAI API + llama-2-70b")
    print("="*60)

    api_key = os.getenv("TOGETHER_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_name="meta-llama/Llama-2-70b-chat-hf",
        deploy_type=ModelDeployType.API,
        api_key=api_key,
        api_url="https://api.together.xyz/v1",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/api_llama70b",
        use_zh_prompt=False,
        sample_size=50
    )


def example_compare_models():
    """示例6: 对比多个模型"""
    print("="*60)
    print("示例6: 对比多个模型")
    print("="*60)

    models = [
        ("llama3", ModelDeployType.OLLAMA),
        ("qwen:7b", ModelDeployType.OLLAMA),
        ("qwen:14b", ModelDeployType.OLLAMA),
    ]

    for model_name, deploy_type in models:
        print(f"\n正在测试: {model_name}...")

        config = LLMConfig(
            model_name=model_name,
            deploy_type=deploy_type,
            ollama_url="http://localhost:11434" if deploy_type == ModelDeployType.OLLAMA else None,
            temperature=0.0,
            max_tokens=100
        )

        output_dir = f"./results/comparison/{model_name.replace(':', '_')}"

        run_inference(
            model_config=config,
            data_path="../summary_nli_hallucination_dataset.xlsx",
            output_dir=output_dir,
            use_zh_prompt=False,
            sample_size=100
        )


def example_full_test():
    """示例7: 使用全部测试数据"""
    print("="*60)
    print("示例7: 使用全部900个测试样本")
    print("="*60)

    config = LLMConfig(
        model_name="llama3",
        deploy_type=ModelDeployType.OLLAMA,
        ollama_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/llama3_full",
        use_zh_prompt=False,
        sample_size=None  # None表示使用全部数据
    )


def example_chinese_prompts():
    """示例8: 使用中文提示词"""
    print("="*60)
    print("示例8: 使用中文提示词")
    print("="*60)

    config = LLMConfig(
        model_name="qwen:7b",
        deploy_type=ModelDeployType.OLLAMA,
        ollama_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=100
    )

    run_inference(
        model_config=config,
        data_path="../summary_nli_hallucination_dataset.xlsx",
        output_dir="./results/qwen_chinese",
        use_zh_prompt=True,
        sample_size=100
    )


def create_comparison_report():
    """创建对比报告"""
    import json
    import pandas as pd
    from pathlib import Path

    print("="*60)
    print("生成模型对比报告")
    print("="*60)

    results_dir = Path("./results/comparison")
    if not results_dir.exists():
        print("对比结果目录不存在，请先运行多个模型的推理")
        return

    # 收集所有结果
    all_results = {}
    for result_file in results_dir.glob("*/llm_results.json"):
        model_name = result_file.parent.name
        with open(result_file, 'r', encoding='utf-8') as f:
            all_results[model_name] = json.load(f)

    # 创建对比表格
    comparison_data = []
    for model_name, result in all_results.items():
        metrics = result['detailed_metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Macro-Precision': f"{metrics['macro_precision']:.4f}",
            'Macro-Recall': f"{metrics['macro_recall']:.4f}",
            'Macro-F1': f"{metrics['macro_f1']:.4f}",
            'Hall-Precision': f"{metrics['hallucination']['precision']:.4f}",
            'Hall-Recall': f"{metrics['hallucination']['recall']:.4f}",
            'Hall-F1': f"{metrics['hallucination']['f1_score']:.4f}",
        })

    comparison_df = pd.DataFrame(comparison_data)

    # 保存对比结果
    output_file = "./results/comparison_report.xlsx"
    comparison_df.to_excel(output_file, index=False)

    print(f"\n对比报告已保存到: {output_file}")
    print("\n对比结果:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    import sys

    # 使用方式：
    # python config_examples.py 1    # 运行示例1
    # python config_examples.py all  # 运行所有示例
    # python config_examples.py compare  # 对比多个模型

    if len(sys.argv) > 1:
        example = sys.argv[1]

        if example == "1":
            example_ollama_llama3()
        elif example == "2":
            example_ollama_qwen()
        elif example == "3":
            example_huggingface_llama3()
        elif example == "4":
            example_api_qwen()
        elif example == "5":
            example_api_openai_llama()
        elif example == "6":
            example_compare_models()
        elif example == "7":
            example_full_test()
        elif example == "8":
            example_chinese_prompts()
        elif example == "compare":
            example_compare_models()
        elif example == "report":
            create_comparison_report()
        elif example == "all":
            print("开始运行所有示例...\n")
            try:
                example_ollama_llama3()
                print("\n" + "="*60 + "\n")
                example_ollama_qwen()
                print("\n" + "="*60 + "\n")
                example_chinese_prompts()
                print("\n完成所有示例！")
            except Exception as e:
                print(f"执行出错: {e}")
        else:
            print("未知示例")
            print_usage()
    else:
        print_usage()


def print_usage():
    """打印使用说明"""
    print("""
    使用方式：

    python config_examples.py <example_number>

    可用示例：
    1  - Ollama + llama3
    2  - Ollama + qwen:7b
    3  - HuggingFace + llama3
    4  - 阿里云API + Qwen
    5  - OpenAI API + llama-2-70b
    6  - 对比多个模型
    7  - 使用全部测试数据
    8  - 使用中文提示词

    compare - 运行所有模型对比
    report  - 生成对比报告
    all     - 运行所有示例

    例子：
    python config_examples.py 1
    python config_examples.py compare
    python config_examples.py report
    """)
