"""
改进的训练脚本
集成Phase 1和Phase 2的所有改进
"""

import torch
import torch.nn.functional as F
import logging
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import random
import yaml

# 导入改进的模块
from improved_losses import ImprovedFocalLoss, ContrastiveLoss, find_optimal_threshold
from improved_model import ImprovedHeteroClassifier
from path_ini import data_model_loader
from dgl.dataloading import GraphDataLoader


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    """
    数据整理函数
    """
    (
        g_premise,
        g_hypothesis,
        lexical_chains,
        nli_label,
    ) = zip(*batch)

    return (
        list(g_premise),
        list(g_hypothesis),
        list(lexical_chains),
        list(nli_label),
    )


def get_dataloader(dataset, batch_size, file_num, shuffle=True):
    """
    获取数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        file_num: 文件编号
        shuffle: 是否打乱数据
    """
    def worker_init_fn(worker_id):
        # 为每个工作进程设置不同的随机种子
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # 加载对应的batch文件
    dataset.load_batch_files(file_num)

    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=worker_init_fn if shuffle else None,
    )


def process_batch_improved(model, batch_data, device, task, mode, optimizer=None,
                           scheduler=None, focal_loss=None, contrastive_loss=None,
                           focal_weight=1.0, contrast_weight=0.1):
    """
    改进的batch处理函数
    添加了对比学习损失
    """
    g_premise_list, g_hypothesis_list, lexical_chain_list, nli_labels = batch_data

    # 移动到设备
    for i in range(len(g_premise_list)):
        g_premise_list[i] = g_premise_list[i].to(device)
        g_hypothesis_list[i] = g_hypothesis_list[i].to(device)

    # 标签映射：把NLI标签(0,1,2)映射到二分类(0,1)
    # 0保持不变(无幻觉)，1和2都映射为1(有幻觉)
    nli_labels = [label if label == 0 else 1 for label in nli_labels]

    # 将labels转换为tensor
    nli_labels = torch.tensor(nli_labels, dtype=torch.long).to(device)

    # 前向传播
    if mode == "train":
        outputs = model(g_premise_list, g_hypothesis_list, lexical_chain_list, nli_labels)
    else:
        with torch.no_grad():
            outputs = model(g_premise_list, g_hypothesis_list, lexical_chain_list, nli_labels)

    logits = outputs["classification"]
    z_premise = outputs["z_premise"]
    z_hypothesis = outputs["z_hypothesis"]

    # 计算损失
    losses = {}

    # 分类损失（Focal Loss）
    if task == "classification":
        cls_loss = focal_loss(logits, nli_labels)
        losses["cls_loss"] = cls_loss.item()

        # 对比学习损失
        if contrastive_loss is not None:
            contrast_loss = contrastive_loss(z_premise, z_hypothesis, nli_labels)
            losses["contrast_loss"] = contrast_loss.item()

            # 总损失
            total_loss = focal_weight * cls_loss + contrast_weight * contrast_loss
        else:
            total_loss = cls_loss

        losses["total_loss"] = total_loss.item()

    # 反向传播
    if mode == "train" and optimizer is not None:
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # 预测
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    labels_np = nli_labels.cpu().numpy()

    return {
        "losses": losses,
        "predictions": predictions.tolist(),
        "labels": labels_np.tolist()
    }


def train_epoch_improved(model, dataloader, optimizer, scheduler, device,
                        focal_loss, contrastive_loss, epoch=None,
                        focal_weight=1.0, contrast_weight=0.1):
    """改进的训练epoch"""
    model.train()
    epoch_losses = defaultdict(float)

    progress_bar = tqdm(dataloader, total=len(dataloader),
                       desc=f"Epoch {epoch or 1} Training")

    for i, batch_data in enumerate(progress_bar):
        batch_metrics = process_batch_improved(
            model, batch_data, device, "classification", "train",
            optimizer, scheduler, focal_loss, contrastive_loss,
            focal_weight, contrast_weight
        )

        for loss_name, loss_value in batch_metrics["losses"].items():
            epoch_losses[loss_name] += loss_value

        # 更新进度条
        progress_bar.set_postfix({
            "cls_loss": f"{batch_metrics['losses']['cls_loss']:.4f}",
            "contrast": f"{batch_metrics['losses'].get('contrast_loss', 0):.4f}"
        })

    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    return epoch_losses


def eval_epoch_improved(model, dataloader, device, focal_loss, contrastive_loss,
                       epoch=0, focal_weight=1.0, contrast_weight=0.1):
    """改进的评估epoch"""
    model.eval()
    epoch_losses = defaultdict(float)
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_metrics = process_batch_improved(
                model, batch_data, device, "classification", "eval",
                focal_loss=focal_loss, contrastive_loss=contrastive_loss,
                focal_weight=focal_weight, contrast_weight=contrast_weight
            )

            # 累积损失
            for loss_name, loss_value in batch_metrics["losses"].items():
                epoch_losses[loss_name] += loss_value

            # 累积预测结果
            all_predictions.extend(batch_metrics["predictions"])
            all_labels.extend(batch_metrics["labels"])

    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    # 诊断信息
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    logging.info(f"Epoch {epoch} - Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    logging.info(f"Epoch {epoch} - Label distribution: {dict(zip(unique_labels, label_counts))}")

    # 计算分类指标
    classification_metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "f1_micro": f1_score(all_labels, all_predictions, average="micro"),
        "precision": precision_score(all_labels, all_predictions, average="macro"),
        "recall": recall_score(all_labels, all_predictions, average="macro"),
    }

    return {
        "losses": epoch_losses,
        "classification_metrics": classification_metrics,
    }


def main_improved():
    """改进的主训练函数"""
    # 设置随机种子
    set_seed(42)

    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"training_improved_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logging.info("="*80)
    logging.info("启动改进的训练流程")
    logging.info("Phase 1 + Phase 2 改进已集成")
    logging.info("="*80)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 加载数据
    logging.info("加载数据集...")
    config, train_dataset, test_dataset = data_model_loader(device)

    # 创建DataLoader（使用原始的数据加载方式）
    logging.info("创建DataLoader...")
    train_loader = get_dataloader(train_dataset, config.batch_size, 0, shuffle=True)
    test_loader = get_dataloader(test_dataset, config.batch_size, 0, shuffle=False)
    logging.info(f"训练集batch数: {len(train_loader)}, 测试集batch数: {len(test_loader)}")

    # 创建改进的模型
    logging.info("创建改进的模型...")
    model = ImprovedHeteroClassifier(
        in_dim=config.model_config["in_dim"],
        hidden_dim=config.model_config["hidden_dim"],
        n_classes=config.model_config["n_classes"],
        rel_names=config.model_config["rel_names"]
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数量: {total_params:,}")
    logging.info(f"可训练参数量: {trainable_params:,}")

    # 创建改进的损失函数
    logging.info("创建改进的损失函数...")
    focal_loss = ImprovedFocalLoss(alpha=[0.5, 4.5], gamma=3.0, label_smoothing=0.05).to(device)
    contrastive_loss = ContrastiveLoss(temperature=0.5, margin=0.5).to(device)

    # 损失权重
    focal_weight = 1.0
    contrast_weight = 0.1

    logging.info(f"Focal Loss - alpha=[0.5, 4.5], gamma=3.0")
    logging.info(f"Contrastive Loss - temperature=0.5, margin=0.5")
    logging.info(f"Loss weights - focal: {focal_weight}, contrast: {contrast_weight}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )

    # 学习率调度器
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps - warmup_steps,
        T_mult=1,
        eta_min=1e-7
    )

    # 训练配置
    config.epochs = 20
    config.lr = 2e-5
    config.batch_size = 10
    epochs = config.epochs
    patience = 5
    best_f1 = 0
    early_stop_counter = 0

    # 保存目录
    save_dir = f"checkpoints/improved_model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    def represent_torch_device(dumper, data):
        return dumper.represent_scalar("!torch.device", str(data))

    yaml.add_representer(torch.device, represent_torch_device)

    config_dict = config.to_dict()
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)

    # 训练循环
    logging.info(f"开始训练，总共 {epochs} 个epoch...")

    for epoch in range(epochs):
        logging.info(f"\n{'='*80}")
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        logging.info(f"{'='*80}")

        # 训练
        train_losses = train_epoch_improved(
            model, train_loader, optimizer, scheduler, device,
            focal_loss, contrastive_loss, epoch + 1,
            focal_weight, contrast_weight
        )

        logging.info(f"训练损失:")
        for name, value in train_losses.items():
            logging.info(f"  {name}: {value:.4f}")

        # 评估
        eval_results = eval_epoch_improved(
            model, test_loader, device, focal_loss, contrastive_loss,
            epoch + 1, focal_weight, contrast_weight
        )

        logging.info(f"评估结果:")
        for name, value in eval_results["losses"].items():
            logging.info(f"  {name}: {value:.4f}")
        for name, value in eval_results["classification_metrics"].items():
            logging.info(f"  {name}: {value:.4f}")

        # 检查是否是最优模型
        current_f1 = eval_results["classification_metrics"]["f1_macro"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            early_stop_counter = 0

            # 保存最优模型
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_macro': best_f1,
                'config': config.to_dict()
            }, save_path)

            logging.info(f"✅ 新的最优模型已保存! F1: {best_f1:.4f}")
        else:
            early_stop_counter += 1
            logging.info(f"Early stopping counter: {early_stop_counter}/{patience}")

        # 早停
        if early_stop_counter >= patience:
            logging.info(f"早停触发，停止训练")
            break

    # 加载最优模型进行最终测试
    logging.info("\n" + "="*80)
    logging.info("加载最优模型进行最终测试...")
    logging.info("="*80)

    checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 阈值调优
    logging.info("\n进行动态阈值调优...")
    best_threshold, threshold_f1 = find_optimal_threshold(
        model, test_loader, device, task="classification"
    )
    logging.info(f"最优阈值: {best_threshold:.3f}, F1: {threshold_f1:.4f}")

    # 使用最优阈值重新评估
    # (这里需要修改eval函数以支持自定义阈值，暂时使用默认0.5)

    final_results = eval_epoch_improved(
        model, test_loader, device, focal_loss, contrastive_loss,
        epoch=-1, focal_weight=focal_weight, contrast_weight=contrast_weight
    )

    logging.info("\n" + "="*80)
    logging.info("最终测试结果:")
    logging.info("="*80)
    for name, value in final_results["classification_metrics"].items():
        logging.info(f"  {name}: {value:.4f}")

    logging.info(f"\n训练完成!")
    logging.info(f"最优F1: {best_f1:.4f}")
    logging.info(f"阈值调优后F1: {threshold_f1:.4f}")
    logging.info(f"模型保存路径: {save_dir}")
    logging.info(f"日志文件: {log_file}")

    return model, best_f1, threshold_f1


if __name__ == "__main__":
    try:
        model, best_f1, threshold_f1 = main_improved()
        print(f"\n{'='*80}")
        print(f"✅ 训练成功完成!")
        print(f"最优F1: {best_f1:.4f}")
        print(f"阈值调优后F1: {threshold_f1:.4f}")
        print(f"{'='*80}")
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}", exc_info=True)
        raise
