#!/usr/bin/env python3
"""
生成 Qwen 批量推理 JSONL 文件
用于阿里云 Dashscope 平台的批量推理功能
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List
from hallucination_prompts import get_prompt

# 配置
DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"
CACHE_FILE = "test_evaluation/rst_cache.pkl"
OUTPUT_DIR = "batch_inference_jsonl"

# 模型列表
MODELS = [
    "qwen3.5-plus",
    "qwen3-max",
    "deepseek-r1",
    "deepseek-v3",
    "qwq-plus"
]

def generate_batch_jsonl(model_name: str, prompt_type: int, test_data: List[Dict], rst_cache: Dict):
    """
    生成单个模型的 JSONL 批量推理文件

    Args:
        model_name: 模型名称
        prompt_type: Prompt 类型（1, 2, 3）
        test_data: 测试数据
        rst_cache: RST 缓存
    """
    output_file = Path(OUTPUT_DIR) / f"{model_name}-type{prompt_type}.jsonl"

    print(f"\n生成: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(test_data):
            sample_id = sample.get('id', idx)
            context = sample.get('context', '')
            output = sample.get('output', '')

            # 从缓存获取 RST 结果
            if sample_id not in rst_cache:
                print(f"  ⚠️  样本 {sample_id} 不在缓存中，跳过")
                continue

            cache_data = rst_cache[sample_id]
            if isinstance(cache_data, dict):
                rst_pred = cache_data['rst_pred']
                rst_prob = cache_data['rst_prob']
                # 优化后的Type 3：高置信度闸门 - 只保留置信度 > 0.7 的EDU（过滤噪音）
                all_edus = cache_data.get('hallucination_edus', [])
                hallucination_edus = [
                    edu for edu in all_edus
                    if isinstance(edu, dict) and edu.get('prob', 0) >= 0.7
                ]
                # 负向引导：如果全局置信度不明确，告诉LLM要自主判别
                unclear_global = 0.4 <= rst_prob <= 0.6
                need_independent_judgment = unclear_global
            else:
                # 向后兼容旧的元组格式
                _, rst_pred, rst_prob = cache_data[:3]
                hallucination_edus = []
                unclear_global = 0.4 <= rst_prob <= 0.6
                need_independent_judgment = unclear_global

            # 构建 Prompt
            try:
                if prompt_type == 1:
                    prompt = get_prompt(1, context, output)
                elif prompt_type == 2:
                    # 如果全局置信度不明确，加入独立判别提示
                    base_prompt = get_prompt(
                        2, context, output,
                        rst_hallucination=bool(rst_pred),
                        rst_prob=float(rst_prob)
                    )
                    if unclear_global:
                        base_prompt = base_prompt.replace(
                            "【参考信息】",
                            "【参考信息】\n⚠️ 注意：系统全局置信度不明确，请完全独立判别\n"
                        )
                    prompt = base_prompt
                elif prompt_type == 3:
                    prompt = get_prompt(
                        3, context, output,
                        rst_hallucination=bool(rst_pred),
                        rst_prob=float(rst_prob),
                        hallucination_edus=hallucination_edus  # 已筛选过
                    )
                else:
                    print(f"  ✗ 无效的 prompt_type: {prompt_type}")
                    continue

                # 构建 JSONL 行
                jsonl_item = {
                    "custom_id": f"{sample_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_tokens": 512
                    }
                }

                # 写入 JSONL（每行一个 JSON 对象，无缩进）
                f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"  ✗ 样本 {sample_id} 处理失败: {str(e)[:100]}")
                continue

    # 计数行数
    line_count = sum(1 for _ in open(output_file))
    print(f"  ✓ 已生成 {line_count} 行")
    return line_count

def main():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("="*80)
    print("生成 Qwen 批量推理 JSONL 文件")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    with open(DATA_PATH, 'r') as f:
        all_data = json.load(f)
    test_data = [s for s in all_data if s.get('dataset') == 'test']
    print(f"✓ 加载了 {len(test_data)} 个测试样本")

    # 加载缓存
    if not Path(CACHE_FILE).exists():
        print(f"✗ 缓存文件不存在: {CACHE_FILE}")
        print("请先运行: python evaluate_test_set.py --cache-rst")
        return

    with open(CACHE_FILE, 'rb') as f:
        rst_cache = pickle.load(f)
    print(f"✓ 加载了 {len(rst_cache)} 个缓存结果")

    # 生成所有模型和所有 Prompt 类型的文件
    print("\n" + "="*80)
    print("生成 JSONL 文件")
    print("="*80)

    total_lines = 0
    files_generated = []

    for model in MODELS:
        print(f"\n【模型: {model}】")
        for prompt_type in [3]:
            try:
                line_count = generate_batch_jsonl(model, prompt_type, test_data, rst_cache)
                total_lines += line_count
                files_generated.append(f"{model}-type{prompt_type}.jsonl")
            except Exception as e:
                print(f"  ✗ 生成失败: {str(e)}")

    # 打印总结
    print("\n" + "="*80)
    print("生成完成")
    print("="*80)
    print(f"\n✓ 已生成 {len(files_generated)} 个文件")
    print(f"✓ 总计 {total_lines} 行数据")
    print(f"✓ 保存位置: {Path(OUTPUT_DIR).absolute()}")

    print("\n生成的文件列表:")
    for filename in files_generated:
        filepath = Path(OUTPUT_DIR) / filename
        filesize = filepath.stat().st_size / 1024 / 1024  # MB
        print(f"  • {filename} ({filesize:.2f} MB)")

    # 打印使用指南
    print("\n" + "="*80)
    print("后续步骤")
    print("="*80)
    print("""
1. 上传文件到阿里云
   - 登录 https://dashscope.aliyun.com
   - 进入 "批量推理" 功能
   - 选择对应的 JSONL 文件上传

2. 启动批量推理
   - 选择要推理的文件
   - 点击"开始推理"

3. 等待结果
   - 批量推理通常需要几小时
   - 完成后下载结果 JSON 文件

4. 后处理
   - 使用 evaluate_batch_results.py 处理结果
   - 计算性能指标

参考命令（获取推理状态）:
  curl -H "Authorization: Bearer YOUR_API_KEY" \\
       https://dashscope.aliyuncs.com/api/v1/batch/jobs

文件位置:
  {Path(OUTPUT_DIR).absolute()}
""")

if __name__ == "__main__":
    main()
