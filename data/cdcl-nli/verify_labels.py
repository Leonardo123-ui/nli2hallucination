"""
验证标签映射是否正确
"""

import torch
from train_improved import get_dataloader, collate_fn
from path_ini import data_model_loader

print("="*80)
print("验证标签映射...")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config, train_dataset, test_dataset = data_model_loader(device)

# 创建DataLoader
train_loader = get_dataloader(train_dataset, batch_size=10, file_num=0, shuffle=True)

# 获取一个batch
for batch_data in train_loader:
    g_premise_list, g_hypothesis_list, lexical_chain_list, nli_labels = batch_data

    print(f"\n原始标签: {nli_labels}")

    # 应用映射
    nli_labels_mapped = [label if label == 0 else 1 for label in nli_labels]

    print(f"映射后标签: {nli_labels_mapped}")

    # 统计
    from collections import Counter
    print(f"\n原始标签分布: {Counter(nli_labels)}")
    print(f"映射后标签分布: {Counter(nli_labels_mapped)}")

    # 验证映射后的标签都在[0,1]范围内
    assert all(label in [0, 1] for label in nli_labels_mapped), "标签映射失败！"

    print("\n✅ 标签映射正确！所有标签都在[0,1]范围内。")
    break

print("\n" + "="*80)
print("现在可以安全地开始训练了！")
print("  CUDA_VISIBLE_DEVICES=0 python train_improved.py")
print("="*80)
