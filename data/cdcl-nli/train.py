##
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import dgl
import json
import base64
import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast
import yaml
from torch.optim import AdamW
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from build_base_graph_extract import (
    ExplainableHeteroClassifier,
    save_texts_to_json,
)
from cal_scores import (

    is_best_model,
)

from path_ini import data_model_loader

from collections import defaultdict
import torch.multiprocessing as mp

# 多进程配置
from tqdm import tqdm 
import random
import logging

def set_seed(seed=42):
    """
    设置所有随机种子以确保结果可复现

    Args:
        seed (int): 随机种子值，默认为42
    """
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(seed)  # PyTorch的GPU随机种子
    dgl.random.seed(seed)  # DGL的随机种子

    # 设置CUDA的随机种子生成器
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动调优功能


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Focal Loss Implementation ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for each class to handle imbalance
        gamma: Focusing parameter for modulating loss (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Apply label smoothing to prevent overconfidence
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # 增强型参数，提高对类别不平衡的处理
        self.use_adaptive_alpha = True

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class labels
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
            ce_loss = -(targets_one_hot * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities and compute focal weights
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
                alpha_t = alpha.gather(0, targets)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===================================================================
# Focal loss with enhanced class weights to handle imbalance
# alpha=[0.9, 3.6]: 更激进的少数类权重
# gamma=2.5: 更强的难样本关注
# label_smoothing=0.1: 防止过度自信
criterion = FocalLoss(alpha=[0.9, 3.6], gamma=2.5, label_smoothing=0.1)

rel_names_long = [
    "Temporal",
    "TextualOrganization",
    "Joint",
    "Topic-Comment",
    "Comparison",
    "Condition",
    "Contrast",
    "Evaluation",
    "Topic-Change",
    "Summary",
    "Manner-Means",
    "Attribution",
    "Cause",
    "Background",
    "Enablement",
    "Explanation",
    "Same-Unit",
    "Elaboration",
    "span",
    "lexical",  # 词汇链
]

# 优化的中间关系列表（15个），平衡信息保留和噪声减少）
rel_names_medium = [
    "Temporal",
    "Joint",
    "Topic-Comment",
    "Condition",
    "Contrast",
    "Evaluation",
    "Summary",
    "Manner-Means",
    "Attribution",
    "Cause",
    "Background",
    "Enablement",
    "Explanation",
    "Elaboration",
    "lexical",
]

rel_names_short = ["Temporal",
                   "Summary",
                   "Condition",
                   "Contrast",
                   "Cause",
                   "Background",
                   "Elaboration",
                   "Explanation",
                   "lexical",
                   ]


def get_dataloader(dataset, batch_size, file_num,  shuffle=True):
    """
    获取数据加载器

    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
    """

    def worker_init_fn(worker_id):
        # 为每个工作进程设置不同的随机种子
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset.load_batch_files(file_num)  # 一个文件
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 设置为0，完全禁用多进程
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=worker_init_fn if shuffle else None,
    )

def save_model(model, path, optimizer=None, scheduler=None, epoch=None, metrics=None):
    """
    保存模型和训练状态

    Args:
        model: 模型实例
        path: 保存路径
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）
        epoch: 当前轮次（可选）
        metrics: 评估指标（可选）
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config if hasattr(model, "config") else None,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }

    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        save_dict["epoch"] = epoch
    if metrics is not None:
        save_dict["metrics"] = metrics

    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 保存模型
    torch.save(save_dict, path)
    logging.info(f"Model saved to {path}")


def load_model(model, path, optimizer=None, scheduler=None):
    """
    加载模型和训练状态

    Args:
        model: 模型实例
        path: 加载路径
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）

    Returns:
        model: 加载后的模型
        optimizer: 加载后的优化器（如果提供）
        scheduler: 加载后的学习率调度器（如果提供）
        epoch: 保存时的轮次
        metrics: 保存时的评估指标
    """
    # 加载保存的字典
    checkpoint = torch.load(path)

    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])

    # 如果保存了配置，并且模型有 config 属性，则加载配置
    if "config" in checkpoint and hasattr(model, "config"):
        model.config = checkpoint["config"]

    # 如果提供了优化器，加载其状态
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 

    # 如果提供了学习率调度器，加载其状态
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


    # 获取保存的轮次和指标
    epoch = checkpoint.get("epoch")
    metrics = checkpoint.get("metrics")

    logging.info(f"Model loaded from {path}")
    return model, optimizer, scheduler, epoch, metrics


# 对node的text进行Base64解码
def decode_text_from_tensor(encoded_tensor):
    # 去除填充的空字节并转换为字节数据
    byte_data = bytes(encoded_tensor.tolist()).rstrip(b"\0")
    # 使用 Base64 解码还原原始文本
    decoded_text = base64.b64decode(byte_data).decode("utf-8")
    return decoded_text


def log_metrics(
    epoch, train_losses, eval_losses, eval_metrics_cli, stage
):
    """
    记录训练和评估指标

    Args:
        epoch: 当前轮次
        train_losses: 训练损失字典
        eval_losses: 评估指标字典
        stage: 训练阶段
    """
    # 构建日志消息
    log_msg = f"Epoch {epoch} ({stage})\n"

    # 添加训练损失
    if train_losses != None:
        log_msg += "Losses:\n"
        for loss_name, loss_value in train_losses.items():
            log_msg += f"  {loss_name}: {loss_value:.4f}\n"

    # 添加评估指标
    log_msg += "Evaluation Metrics:\n"
    for metric_name, metric_value in eval_losses.items():
        log_msg += f"  {metric_name}: {metric_value:.4f}\n"
    for metric_name, metric_value in eval_metrics_cli.items():
        log_msg += f"  {metric_name}: {metric_value:.4f}\n"
    # 记录日志
    logging.info(log_msg)

def get_current_metric(metrics, stage):
    """
    获取当前阶段的主要指标值

    Args:
        metrics: 评估指标字典
        stage: 训练阶段
    """
    if stage == "classification":
        return metrics.get("accuracy", 0)
    elif stage == "generation":
        return metrics.get("bleu", 0)
    else:  # joint
        return metrics.get("accuracy", 0) * 0.6 + metrics.get("bleu", 0) * 0.4

# 数据收集器类，用于跟踪训练过程中的指标
class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metrics_dict):
        """更新指标"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)

    def get_average(self, key):
        """获取特定指标的平均值"""
        values = self.metrics[key]
        return sum(values) / len(values) if values else 0

    def reset(self):
        """重置所有指标"""
        self.metrics.clear()

    def get_all_averages(self):
        """获取所有指标的平均值"""
        return {key: self.get_average(key) for key in self.metrics}


# ==================== Graph Data Augmentation ====================
class GraphAugmentation:
    """
    Graph data augmentation for training.
    Includes: edge dropout and node feature perturbation.
    """
    def __init__(self, edge_drop_prob=0.1, node_perturb_ratio=0.05):
        """
        Args:
            edge_drop_prob: Probability of dropping an edge (default: 0.1)
            node_perturb_ratio: Ratio of Gaussian noise to add to node features (default: 0.05)
        """
        self.edge_drop_prob = edge_drop_prob
        self.node_perturb_ratio = node_perturb_ratio

    def edge_dropout(self, graph):
        """
        Randomly drop edges from the graph.

        Args:
            graph: DGL heterograph
        Returns:
            Augmented graph with some edges dropped
        """
        new_graph_data = {}

        for etype in graph.canonical_etypes:
            src, dst = graph.edges(etype=etype)
            num_edges = len(src)

            if num_edges == 0:
                new_graph_data[etype] = ([], [])
                continue

            # Create dropout mask
            keep_mask = torch.rand(num_edges) > self.edge_drop_prob

            # Keep at least one edge per edge type to maintain graph connectivity
            if keep_mask.sum() == 0:
                keep_mask[0] = True

            new_src = src[keep_mask]
            new_dst = dst[keep_mask]

            new_graph_data[etype] = (new_src.tolist(), new_dst.tolist())

        # Create new graph
        new_graph = dgl.heterograph(
            new_graph_data,
            num_nodes_dict={"node": graph.num_nodes()}
        )

        # Copy node features
        for key in graph.ndata.keys():
            new_graph.ndata[key] = graph.ndata[key].clone()

        return new_graph

    def node_feature_perturbation(self, graph):
        """
        Add small Gaussian noise to node features.

        Args:
            graph: DGL heterograph
        Returns:
            Graph with perturbed node features
        """
        if "feat" not in graph.ndata:
            return graph

        features = graph.ndata["feat"]

        # Add Gaussian noise: noise_std = feature_std * perturbation_ratio
        feature_std = features.std()
        noise = torch.randn_like(features) * feature_std * self.node_perturb_ratio

        # Create new graph (shallow copy)
        new_graph = graph.clone()
        new_graph.ndata["feat"] = features + noise

        return new_graph

    def augment(self, graph, apply_edge_drop=True, apply_node_perturb=True):
        """
        Apply multiple augmentations to a graph.

        Args:
            graph: Input graph
            apply_edge_drop: Whether to apply edge dropout
            apply_node_perturb: Whether to apply node feature perturbation
        Returns:
            Augmented graph
        """
        aug_graph = graph

        if apply_edge_drop:
            aug_graph = self.edge_dropout(aug_graph)

        if apply_node_perturb:
            aug_graph = self.node_feature_perturbation(aug_graph)

        return aug_graph

# Initialize global augmentation
graph_augmentor = GraphAugmentation(edge_drop_prob=0.1, node_perturb_ratio=0.05)
# ===================================================================


def collate_fn(batch):
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
        list(nli_label),  # 保持列表形式,
    )


def process_batch(
    model, batch_data, device, task, stage="train", optimizer=None, scheduler=None, accumulation_steps=2, step_now=None
):
    model.train() if stage == "train" else model.eval()
    """处理一个batch的通用逻辑"""
    graph1, graph2, lexical_chain, nli_labels = batch_data

    # Apply graph augmentation during training only
    if stage == "train":
        graph1 = [graph_augmentor.augment(g) for g in graph1]
        graph2 = [graph_augmentor.augment(g) for g in graph2]

    batch_loss = 0
    batch_size = len(graph1)
    # 合并图
    combined_graphs = [
        model.merge_graphs(g_p1, g_p2, lc, rel_names_medium)   # 使用优化的15个关键关系
        for g_p1, g_p2, lc in zip(graph1, graph2, lexical_chain)
    ]

    combined_graphs = dgl.batch(combined_graphs).to(device)
    for g1 in graph1:
        if "node_id" in g1.ndata:
            g1.ndata.pop("node_id")
    for g2 in graph2:
        if "node_id" in g2.ndata:
            g2.ndata.pop("node_id")
    graph1 = dgl.batch(graph1).to(device)
    graph2 = dgl.batch(graph2).to(device)


    # 获取图表示（使用优化的轻量级方法）
    graph_repr = model.get_graph_repr(combined_graphs)
    pre_graph_repr = model.get_graph_repr(graph1)
    hyp_graph_repr = model.get_graph_repr(graph2)

    # 将nli标签转为targets
    # 把nli_labels里面的2改为1，0不变
    nli_labels = [label if label == 0 else 1 for label in nli_labels]
    targets = torch.tensor(nli_labels, dtype=torch.long, device=device)

    batch_metrics = {
        "losses": defaultdict(float),
        "predictions": [],
        "labels": [],
    }

    # 分类任务
    predicted_cli_batch = []
    labels_cli_batch = []
    classification_losses = []

    # 调试：打印形状信息（仅在第一个batch）
    if step_now == 0:
        print(f"DEBUG: pre_graph_repr shape: {pre_graph_repr.shape}")
        print(f"DEBUG: hyp_graph_repr shape: {hyp_graph_repr.shape}")
        print(f"DEBUG: graph_repr shape: {graph_repr.shape}")
        print(f"DEBUG: batch_size: {batch_size}")

    cli_logits = model.classify(pre_graph_repr, hyp_graph_repr, graph_repr)
    cls_loss = criterion(cli_logits, targets)  # Use global focal loss criterion
    predicted = torch.argmax(cli_logits, dim=-1).tolist()
    classification_losses.append(cls_loss)
    predicted_cli_batch.extend(predicted)
    labels_cli_batch.extend(targets.cpu().numpy())

    classification_loss = torch.stack(classification_losses).mean()
    batch_loss = classification_loss

    batch_metrics["losses"]["cls_loss"] += classification_loss.item()
    batch_metrics["predictions"].extend(predicted_cli_batch)
    batch_metrics["labels"].extend(labels_cli_batch)

    # logging.info(f"Batch loss: {batch_loss.item()}")
    if stage == "train" and optimizer is not None:
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Diagnostic: Check gradients (keep for monitoring)
        if step_now == 0:  # Only check first batch
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            logging.info(f"Total gradient norm before clipping: {total_grad_norm:.6f}")
            logging.info(f"Gradient clipping applied with max_norm={max_grad_norm}")

        optimizer.step()
        scheduler.step()

    return batch_metrics




def train_epoch(model, dataloader, optimizer, scheduler, task, device, accumulation_steps=2, epoch=None, save_dir="./"):
    """训练一个epoch"""
    model.train()
    epoch_losses = defaultdict(float)
    
    # 2. 将 dataloader 包装在 tqdm 中
    # total=len(dataloader) 可以让进度条知道总步数
    # desc 是进度条的描述信息
    progress_bar = tqdm(
        dataloader, 
        total=len(dataloader), 
        desc=f"Epoch {epoch or 1} Training" # 使用 epoch 编号作为描述
    )

    # 3. 遍历 progress_bar 而不是 dataloader
    for i, batch_data in enumerate(progress_bar): 
        
        batch_metrics = process_batch(
            model, batch_data, device, task, "train", optimizer, scheduler, accumulation_steps, i
        )
        
        for loss_name, loss_value in batch_metrics["losses"].items():
            epoch_losses[loss_name] += loss_value
            
            # 4. (可选但推荐) 在进度条上实时显示当前批次的损失
            # 注意：这里需要确保 loss_value 是一个Python数值（float），而不是Tensor
            if isinstance(loss_value, torch.Tensor):
                display_value = loss_value.item()
            else:
                display_value = loss_value
            
            progress_bar.set_postfix({loss_name: f"{display_value:.4f}"})

    # 循环结束后，i 仍然是最后一个索引
    print("end of one epoch, ", "steps : ", i + 1) # i是从0开始的，所以总步数是 i + 1
    
    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    return epoch_losses

def eval_epoch(model, dataloader, task, device, epoch=0, save_dir="./"):
    """评估一个epoch"""
    model.eval()
    epoch_losses = defaultdict(float)
    all_predictions = []
    all_labels = []
    classification_metrics = {}

    with torch.no_grad():
        for batch_data in dataloader:
            batch_metrics = process_batch(model, batch_data, device, task, "eval")

            # 累积损失
            for loss_name, loss_value in batch_metrics["losses"].items():
                epoch_losses[loss_name] += loss_value

            # 累积预测结果
            all_predictions.extend(batch_metrics["predictions"])
            all_labels.extend(batch_metrics["labels"])

    # 存储预测结果和真实结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 计算平均损失
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)

    # 诊断信息：检查预测和标签的分布
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    logging.info(f"Epoch {epoch} - Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    logging.info(f"Epoch {epoch} - Label distribution: {dict(zip(unique_labels, label_counts))}")

    # 计算分类指标
    classification_metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_macro_cli": f1_score(all_labels, all_predictions, average="macro"),
        "f1_micro_cli": f1_score(all_labels, all_predictions, average="micro"),
        "precision": precision_score(all_labels, all_predictions, average="macro"),
        "recall": recall_score(all_labels, all_predictions, average="macro"),
        "specificity": recall_score(all_labels, all_predictions, average="macro", pos_label=1),
    }
    return {
        "losses": epoch_losses,
        "classification_metrics": classification_metrics,
    }


def represent_torch_device(dumper, data):
    return dumper.represent_scalar("!torch.device", str(data))


def main():
    """
    主训练函数
    """
    # 设置随机种子
    set_seed(42)  # 可以根据需要修改种子值
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )

    try:

        # 初始化模型和数据集
        logging.info(f"Using device: {device}")

        config, train_dataset, test_dataset = data_model_loader(device)
        config.device = device
        config.save_dir = "checkpoints/experiment_6"
        config.mode = "train"
        config.stage = "classification"  # classification / generation / joint
        config.epochs = 25  # Increased from 20 for better convergence with focal loss
        config.lr = 2e-5  # Conservative increase from 1e-5 to 2e-5
        config.batch_size = 10
        stage = config.stage
        config.eval_interval = 1
        
        #   888 = 296 * 3 origin train data
        # all_train_data: 22200 = 7400 * 3 ; num_files : 20 
        
        train_loader = get_dataloader(train_dataset, config.batch_size, 0)
        test_loader = get_dataloader(test_dataset, config.batch_size, 0, shuffle=False)
        print(len(train_loader), len(test_loader))  # 370 124 124  batch size = 15
        # 保存配置
        os.makedirs(config.save_dir, exist_ok=True)
        yaml.add_representer(torch.device, represent_torch_device)

        with open(os.path.join(config.save_dir, f"config-{stage}.yaml"), "w") as f:
            yaml.dump(config.to_dict(), f)
        model = ExplainableHeteroClassifier(
            in_dim=config.model_config["in_dim"],
            hidden_dim=config.model_config["hidden_dim"],
            n_classes=config.model_config["n_classes"],
            rel_names=rel_names_long,  # 模型支持所有关系，避免KeyError
            device=device,
        ).to(device)
        # AdamW optimizer with enhanced regularization
        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,  # Will be 2e-5 from config
            weight_decay=1e-3,  # 增加正则化（从 1e-4）
            eps=1e-8,
            betas=(0.9, 0.999),
            amsgrad=True,  # 使用 AMSGrad 变体，更稳定的收敛
        )

        num_training_steps = config.total_steps
        logging.info(f"Total training steps: {num_training_steps}")
        num_warmup_steps = int(num_training_steps * 0.25)  # 增加预热到 25%（从 20%）
        logging.info(f"Total warmup steps: {num_warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
        # 加载预训练模型
        if os.path.exists(os.path.join(config.save_dir, f"best_{stage}_model.pt")):
            
            print("Loading pre-trained model...")
            model, optimizer, scheduler, start_epoch, metrics = load_model(model, os.path.join(config.save_dir, f"best_{stage}_model.pt"), optimizer, scheduler)
            print(metrics)
        else:
            # 设置优化器和调度器
            
            start_epoch = 0


        # 初始化指标跟踪器
        metrics_tracker = MetricsTracker()

        # 训练循环

        logging.info(f"Starting {config.mode}  ==  {stage} task...")
        best_classification_f1 = 0
        best_joint_f1 = 0
        for epoch in range(start_epoch, config.epochs):
            accumulation_steps = 2  # 4
            logging.info(f"Epoch {epoch} starting")
            train_losses = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                task=config.stage,
                device=device,
                accumulation_steps=accumulation_steps,
                epoch=epoch,
                save_dir=config.save_dir,
            )
            logging.info(f"Epoch {epoch} finished\ntraining losses: {train_losses}")
          
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Current learning rate: {current_lr:.2e}")

            # 诊断：检查模型参数是否在变化
            first_param = next(model.parameters())
            logging.info(f"First model parameter norm: {first_param.norm().item():.6f}")

            if (epoch+1) % config.eval_interval != 0:
                metrics_tracker.update(
                    {
                        **{f"train_{k}": v for k, v in train_losses.items()}
                    }
                )
                
                continue
            
            # 评估
            eval_results = eval_epoch(
                model, test_loader, task=config.stage, device=device, epoch=epoch, save_dir=config.save_dir
            )           

            # 记录指标
            metrics_tracker.update(
                {
                    **{f"train_{k}": v for k, v in train_losses.items()},
                    **{f"eval_{k}": v for k, v in eval_results["losses"].items()},
      
                }
            )
            torch.cuda.empty_cache()
            # 记录日志
            log_metrics(
                epoch,
                train_losses,
                eval_results["losses"],
                eval_results["classification_metrics"],
                stage,
            )
            if stage == "classification" and is_best_model(
                eval_results, best_classification_f1, stage
            ):  # 分类任务，用f1指标
                best_classification_f1 = eval_results["classification_metrics"][
                    "f1_macro_cli"
                ]
                model_path = os.path.join(config.save_dir, f"best_{stage}_model.pt")
                save_model(
                    model,
                    model_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=eval_results["classification_metrics"],
                )
                logging.info(
                    f"New best classification model saved with metric: {best_classification_f1:.4f}"
                )
        # 阶段结束，记录最终指标
        stage_metrics = metrics_tracker.get_all_averages()
        logging.info(f"\n{stage} training completed.")
        logging.info("Training average metrics:")
        for metric_name, value in stage_metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")

        # 重置指标跟踪器，准备下一阶段
        metrics_tracker.reset()

        # 训练完成，记录最终结果
        logging.info("\nTraining completed.")

        # 测试
        logging.info("\nTesting...")
        model, _, _, _, _ = load_model(
            model,
            os.path.join(config.save_dir, f"best_{stage}_model.pt"),
            optimizer,
            scheduler,
        )
        test_metrics = eval_epoch(model, test_loader, task=config.stage, device=device, epoch="final", save_dir=config.save_dir)
        logging.info("\nTest metrics:")

        # 记录日志
        log_metrics(
            "final",
            None,
            test_metrics["losses"],
            test_metrics["classification_metrics"],
            stage,
        )

    finally:
        # 清理资源
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
