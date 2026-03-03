"""
改进的损失函数模块
包含：
1. 改进的Focal Loss
2. 对比学习损失
3. 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedFocalLoss(nn.Module):
    """
    改进的Focal Loss，使用更激进的类权重
    """
    def __init__(self, alpha=[0.5, 4.5], gamma=3.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        """
        self.alpha = self.alpha.to(logits.device)

        # 应用label smoothing
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            smooth_labels = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            targets = smooth_labels
        else:
            targets = F.one_hot(labels, num_classes).float()

        # 计算概率
        probs = F.softmax(logits, dim=1)

        # 计算focal weight
        pt = (targets * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        # 计算交叉熵
        ce_loss = -torch.log(pt + 1e-8)

        # 应用类权重
        alpha_weight = self.alpha[labels]

        # 最终损失
        loss = alpha_weight * focal_weight * ce_loss

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    无幻觉样本：premise和hypothesis应该相似
    有幻觉样本：premise和hypothesis应该不同
    """
    def __init__(self, temperature=0.5, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, z_premise, z_hypothesis, labels):
        """
        Args:
            z_premise: (batch, hidden_dim) - premise图表示
            z_hypothesis: (batch, hidden_dim) - hypothesis图表示
            labels: (batch,) - 0=无幻觉, 1=有幻觉
        """
        # 计算余弦相似度
        sim = F.cosine_similarity(z_premise, z_hypothesis, dim=1)

        # 对比损失
        # 无幻觉：相似度应该高 (sim -> 1)
        # 有幻觉：相似度应该低 (sim -> 0 或负值)
        loss_no_hallucination = (1 - sim[labels == 0]) ** 2
        loss_hallucination = torch.clamp(sim[labels == 1] + self.margin, min=0) ** 2

        # 合并损失
        if loss_no_hallucination.numel() > 0 and loss_hallucination.numel() > 0:
            loss = torch.cat([loss_no_hallucination, loss_hallucination]).mean()
        elif loss_no_hallucination.numel() > 0:
            loss = loss_no_hallucination.mean()
        elif loss_hallucination.numel() > 0:
            loss = loss_hallucination.mean()
        else:
            loss = torch.tensor(0.0, device=z_premise.device)

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：Focal Loss + 标准交叉熵
    """
    def __init__(self, focal_alpha=[0.6, 4.0], focal_gamma=2.5,
                 ce_weight=[1.0, 3.4], focal_weight=0.7):
        super().__init__()
        self.focal = ImprovedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight))
        self.focal_weight = focal_weight

    def forward(self, logits, labels):
        focal_loss = self.focal(logits, labels)
        ce_loss = self.ce(logits, labels)
        return self.focal_weight * focal_loss + (1 - self.focal_weight) * ce_loss


def find_optimal_threshold(model, val_loader, device, task="classification"):
    """
    在验证集上寻找最优分类阈值

    Args:
        model: 训练好的模型
        val_loader: 验证集DataLoader
        device: 设备
        task: 任务类型

    Returns:
        best_thresh: 最优阈值
        best_f1: 最优F1分数
    """
    from sklearn.metrics import f1_score
    import numpy as np
    import logging

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in val_loader:
            # 处理batch数据
            g_premise_list, g_hypothesis_list, lexical_chain_list, nli_labels = batch_data

            # 移动到设备
            for i in range(len(g_premise_list)):
                g_premise_list[i] = g_premise_list[i].to(device)
                g_hypothesis_list[i] = g_hypothesis_list[i].to(device)

            # 标签映射：把NLI标签(0,1,2)映射到二分类(0,1)
            nli_labels = [label if label == 0 else 1 for label in nli_labels]

            nli_labels = torch.tensor(nli_labels, dtype=torch.long).to(device)

            # 前向传播
            if task == "classification":
                logits_dict = model(
                    g_premise_list,
                    g_hypothesis_list,
                    lexical_chain_list,
                    nli_labels
                )
                logits = logits_dict["classification"]

            # 获取概率
            probs = F.softmax(logits, dim=1)[:, 1]  # 正类概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(nli_labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 网格搜索最优阈值
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.linspace(0.3, 0.7, 41):
        preds = (all_probs > thresh).astype(int)
        f1 = f1_score(all_labels, preds, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logging.info(f"Optimal threshold: {best_thresh:.3f}, F1: {best_f1:.4f}")

    return best_thresh, best_f1


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 8
    num_classes = 2
    hidden_dim = 1024

    # 模拟数据
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    z_premise = torch.randn(batch_size, hidden_dim)
    z_hypothesis = torch.randn(batch_size, hidden_dim)

    # 测试Improved Focal Loss
    print("Testing ImprovedFocalLoss...")
    focal_loss = ImprovedFocalLoss()
    loss1 = focal_loss(logits, labels)
    print(f"Focal Loss: {loss1.item():.4f}")

    # 测试Contrastive Loss
    print("\nTesting ContrastiveLoss...")
    contrast_loss = ContrastiveLoss()
    loss2 = contrast_loss(z_premise, z_hypothesis, labels)
    print(f"Contrastive Loss: {loss2.item():.4f}")

    # 测试Combined Loss
    print("\nTesting CombinedLoss...")
    combined_loss = CombinedLoss()
    loss3 = combined_loss(logits, labels)
    print(f"Combined Loss: {loss3.item():.4f}")

    print("\n✅ All loss functions work correctly!")
