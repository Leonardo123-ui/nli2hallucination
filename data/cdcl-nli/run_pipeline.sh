#!/bin/bash

# CDCL-NLI 幻觉检测数据处理流水线
# 自动运行数据转换、RST分析、embedding生成和词汇链计算
#
# 使用方法:
#   ./run_pipeline.sh           # 默认使用GPU 0
#   ./run_pipeline.sh 1         # 使用GPU 1
#   ./run_pipeline.sh 2         # 使用GPU 2

set -e  # 遇到错误立即退出

# GPU设置 (默认使用第0张卡)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 检查 Python 环境
check_environment() {
    print_header "检查环境"

    print_info "当前使用GPU: $GPU_ID"

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 python3"
        exit 1
    fi
    print_info "Python: $(python3 --version)"

    # 检查 CUDA
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_info "CUDA: 可用"
        python3 -c "import torch; print('GPU Device:', torch.cuda.get_device_name(0))"
    else
        print_error "CUDA: 不可用（处理会非常慢）"
    fi

    # 检查依赖包
    print_info "检查依赖包..."
    python3 -c "import torch, transformers, pandas, numpy, nltk" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "所有依赖包已安装"
    else
        print_error "缺少依赖包，请运行: pip install torch transformers pandas numpy nltk tqdm"
        exit 1
    fi

    # 检查模型路径
    if [ -d "/mnt/nlp/yuanmengying/nli2hallucination/models/modern-bert_large" ]; then
        print_success "ModernBERT 模型存在"
    else
        print_error "ModernBERT 模型不存在"
        exit 1
    fi

    # 检查原始数据
    if [ -f "../../summary_nli_hallucination_dataset.xlsx" ]; then
        print_success "原始数据文件存在"
    else
        print_error "原始数据文件不存在: ../../summary_nli_hallucination_dataset.xlsx"
        exit 1
    fi

    echo ""
}

# 步骤 1: 数据转换
step1_convert_data() {
    print_header "步骤 1/3: 数据格式转换"

    if [ -f "./data/hallucination_train.json" ] && [ -f "./data/hallucination_test.json" ]; then
        print_info "转换后的数据已存在，跳过..."
        return 0
    fi

    print_info "开始转换幻觉检测数据为 NLI 格式..."

    python3 convert_hallucination_data.py \
        --excel_path ../../summary_nli_hallucination_dataset.xlsx \
        --output_dir ./data \
        --create_sample \
        --sample_size 100

    if [ $? -eq 0 ]; then
        print_success "数据转换完成"
        print_info "训练集: $(wc -l < ./data/hallucination_train.json 2>/dev/null || echo 'N/A') 行"
        print_info "测试集: $(wc -l < ./data/hallucination_test.json 2>/dev/null || echo 'N/A') 行"
    else
        print_error "数据转换失败"
        exit 1
    fi

    echo ""
}

# 步骤 2: 处理数据（小样本测试）
step2_test_sample() {
    print_header "步骤 2/3: 小样本测试（可选）"

    read -p "是否先用小样本测试？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 0
    fi

    print_info "使用 100 个样本进行快速测试..."
    print_info "这将需要约 5-10 分钟"

    # 备份原始脚本
    cp arrange_hallucination_data.py arrange_hallucination_data.py.bak

    # 修改为使用小样本
    sed -i 's|hallucination_train.json|hallucination_train_sample.json|g' arrange_hallucination_data.py
    sed -i 's|hallucination_test.json|hallucination_test_sample.json|g' arrange_hallucination_data.py

    # 运行处理
    python3 arrange_hallucination_data.py

    # 恢复原始脚本
    mv arrange_hallucination_data.py.bak arrange_hallucination_data.py

    if [ $? -eq 0 ]; then
        print_success "小样本测试完成"
    else
        print_error "小样本测试失败"
        exit 1
    fi

    echo ""
}

# 步骤 3: 完整数据处理
step3_process_full_data() {
    print_header "步骤 3/3: 完整数据处理"

    print_info "这将处理所有数据（约 3-4 小时）"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过完整数据处理"
        return 0
    fi

    print_info "开始处理训练集和测试集..."
    print_info "训练集: ~3-4 小时"
    print_info "测试集: ~30-40 分钟"

    # 记录开始时间
    start_time=$(date +%s)

    # 运行处理
    python3 arrange_hallucination_data.py

    # 记录结束时间
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))

    if [ $? -eq 0 ]; then
        print_success "数据处理完成"
        print_info "总耗时: ${hours}小时${minutes}分钟"
    else
        print_error "数据处理失败"
        exit 1
    fi

    echo ""
}

# 查看结果
show_results() {
    print_header "处理结果"

    if [ -d "./data/train" ]; then
        print_info "训练集结果:"
        if [ -f "./data/train/rst_result.jsonl" ]; then
            lines=$(wc -l < ./data/train/rst_result.jsonl)
            print_success "  RST 结果: ${lines} 样本"
        fi
        if [ -f "./data/train/new_rst_result.jsonl" ]; then
            lines=$(wc -l < ./data/train/new_rst_result.jsonl)
            print_success "  重写 RST 结果: ${lines} 样本"
        fi
        if [ -f "./data/train/node_embeddings.npz" ]; then
            size=$(du -h ./data/train/node_embeddings.npz | cut -f1)
            print_success "  节点 embeddings: ${size}"
        fi
    fi

    if [ -d "./data/test" ]; then
        print_info "测试集结果:"
        if [ -f "./data/test/rst_result.jsonl" ]; then
            lines=$(wc -l < ./data/test/rst_result.jsonl)
            print_success "  RST 结果: ${lines} 样本"
        fi
        if [ -f "./data/test/new_rst_result.jsonl" ]; then
            lines=$(wc -l < ./data/test/new_rst_result.jsonl)
            print_success "  重写 RST 结果: ${lines} 样本"
        fi
        if [ -f "./data/test/node_embeddings.npz" ]; then
            size=$(du -h ./data/test/node_embeddings.npz | cut -f1)
            print_success "  节点 embeddings: ${size}"
        fi
    fi

    if [ -f "./data/graph_info/train/lexical_matrixes.pkl" ]; then
        size=$(du -h ./data/graph_info/train/lexical_matrixes.pkl | cut -f1)
        print_success "词汇链矩阵 (训练集): ${size}"
    fi

    if [ -f "./data/graph_info/test/lexical_matrixes.pkl" ]; then
        size=$(du -h ./data/graph_info/test/lexical_matrixes.pkl | cut -f1)
        print_success "词汇链矩阵 (测试集): ${size}"
    fi

    echo ""
    print_info "所有结果保存在: ./data/"
    echo ""
}

# 主函数
main() {
    echo ""
    print_header "CDCL-NLI 幻觉检测数据处理流水线"
    print_info "使用GPU: $GPU_ID"
    echo ""

    # 检查环境
    check_environment

    # 步骤 1: 数据转换
    step1_convert_data

    # 步骤 2: 小样本测试（可选）
    step2_test_sample

    # 步骤 3: 完整数据处理
    step3_process_full_data

    # 显示结果
    show_results

    print_header "处理完成！"
    print_info "查看 README.md 了解如何使用生成的数据"
    echo ""
}

# 运行主函数
main "$@"
