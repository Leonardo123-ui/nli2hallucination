#!/usr/bin/env python3
"""
LLM推理环境检查脚本
检查是否已安装所有必要的依赖和服务
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("✓ Python版本检查")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python版本过低: {version.major}.{version.minor}")
        print("     需要 Python 3.8+")
        return False


def check_package(package_name, import_name=None):
    """检查Python包"""
    if import_name is None:
        import_name = package_name.replace('-', '_')

    try:
        __import__(import_name)
        print(f"  ✅ {package_name}")
        return True
    except ImportError:
        print(f"  ❌ {package_name} (未安装)")
        return False


def check_python_packages():
    """检查必要的Python包"""
    print("\n✓ Python包检查")

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]

    results = []
    for package, import_name in packages:
        results.append(check_package(package, import_name))

    return all(results)


def check_torch_cuda():
    """检查CUDA支持"""
    print("\n✓ CUDA/GPU检查")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用 (GPU: {torch.cuda.get_device_name(0)})")
            print(f"     CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  CUDA不可用（使用CPU会很慢）")
            return False
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False


def check_ollama():
    """检查Ollama服务"""
    print("\n✓ Ollama服务检查")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"  ✅ Ollama服务正在运行")
                print(f"     可用模型: {len(models)}")
                for model in models[:3]:
                    name = model.get('name', 'unknown')
                    print(f"     - {name}")
                return True
            else:
                print("  ⚠️  Ollama运行中，但没有模型")
                print("     运行: ollama pull llama3")
                return False
        else:
            print(f"  ⚠️  Ollama响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama服务未运行")
        print("     启动: ollama serve")
        return False
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False


def check_model_files():
    """检查所需的脚本文件"""
    print("\n✓ 脚本文件检查")

    required_files = [
        "llm_inference.py",
        "config_examples.py",
        "compare_models.py",
    ]

    doc_files = [
        "QUICK_START.md",
        "LLM_INFERENCE_GUIDE.md",
        "LLM_README.md",
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (未找到)")
            all_exist = False

    print("\n  文档文件:")
    for file in doc_files:
        path = Path(file)
        if path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} (可选)")

    return all_exist


def check_data_files():
    """检查数据文件"""
    print("\n✓ 数据文件检查")

    data_path = Path("../summary_nli_hallucination_dataset.xlsx")
    if data_path.exists():
        size = data_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ 测试数据存在 ({size:.2f}MB)")
        return True
    else:
        print(f"  ❌ 测试数据未找到: {data_path}")
        return False


def print_summary(results):
    """打印检查总结"""
    print("\n" + "="*60)
    print("环境检查总结")
    print("="*60)

    all_pass = all(results.values())

    status = "✅ 环境就绪！" if all_pass else "⚠️  存在未满足的要求"
    print(f"\n{status}\n")

    if not all_pass:
        if not results.get('python_version'):
            print("需要升级Python到3.8+")

        if not results.get('packages'):
            print("\n安装依赖包:")
            print("  pip install -r requirements_llm.txt")

        if not results.get('ollama'):
            print("\n要使用Ollama，需要:")
            print("  1. 安装: curl https://ollama.ai/install.sh | sh")
            print("  2. 启动: ollama serve")
            print("  3. 拉取模型: ollama pull llama3")

        if not results.get('files'):
            print("\n脚本文件缺失，请检查目录")

        if not results.get('data'):
            print("\n数据文件缺失，请确保在正确的目录")


def print_quick_start():
    """打印快速开始提示"""
    print("\n" + "="*60)
    print("快速开始")
    print("="*60)

    print("""
1️⃣  快速测试（50个样本）:
  python llm_inference.py --model_name llama3 --sample_size 50

2️⃣  完整测试：
  python llm_inference.py --model_name llama3

3️⃣  使用中文提示词：
  python llm_inference.py --model_name qwen:7b --use_zh_prompt

4️⃣  对比多个模型：
  python config_examples.py compare

5️⃣  查看帮助：
  python llm_inference.py --help

📖 更多信息: 查看 QUICK_START.md 或 LLM_INFERENCE_GUIDE.md
""")


def main():
    """运行所有检查"""
    print("="*60)
    print("LLM推理环境检查")
    print("="*60 + "\n")

    results = {
        'python_version': check_python_version(),
        'packages': check_python_packages(),
        'cuda': check_torch_cuda(),
        'ollama': check_ollama(),
        'files': check_model_files(),
        'data': check_data_files(),
    }

    print_summary(results)

    if all(results.values()):
        print_quick_start()


if __name__ == "__main__":
    main()
