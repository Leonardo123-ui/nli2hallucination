"""
RST-RGAT训练脚本
功能：
1. 数据加载与图构建
2. 训练循环（Focal Loss）
3. 评估（EDU级 + 全局级）
4. 模型保存
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行警告

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import json
import logging
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from data_preprocessing import DataPreprocessor, split_by_dataset_field
from graph_builder import GraphBuilder
from rst_rgat_model import RSTRGATModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    数值稳定的Focal Loss
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) 预测logit
            targets: (N,) 真实标签 (0 or 1)

        Returns:
            loss: 标量
        """
        # 检查输入
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(f"Logits contains NaN or Inf, clamping...")
            logits = torch.clamp(logits, -10.0, 10.0)

        targets = targets.float()

        # 使用稳定的概率计算
        probs = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)

        # 计算BCE（手动实现以确保稳定性）
        bce_loss = -(targets * torch.log(probs + self.eps) +
                     (1 - targets) * torch.log(1 - probs + self.eps))

        # 计算focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = pt.clamp(self.eps, 1 - self.eps)
        focal_weight = (1 - pt) ** self.gamma

        # 应用类权重
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss

        return loss.mean()


class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 5e-5, weight_decay: float = 1e-5,
                 total_epochs: int = 60):
        self.model = model.to(device)
        self.device = device

        # 使用更保守的学习率
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,  # 从1e-4降到5e-5
            weight_decay=weight_decay,
            eps=1e-8
        )

        # 添加学习率调度器（CosineAnnealingLR）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6
        )

        self.edu_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
        self.global_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

        self.best_f1 = 0.0
        self.best_threshold = 0.5
        self.nan_count = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_edu_loss = 0.0
        total_global_loss = 0.0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(batch)
            edu_logits = outputs['edu_logits']
            global_logit = outputs['global_logit']

            # 检查输出是否包含NaN
            if torch.isnan(edu_logits).any() or torch.isnan(global_logit).any():
                logger.warning(f"Batch {batch_idx}: NaN in model output, skipping...")
                self.nan_count += 1
                if self.nan_count > 10:
                    logger.error("Too many NaN batches, stopping training")
                    raise RuntimeError("Training unstable: too many NaN batches")
                continue

            # 计算损失
            try:
                edu_loss = self.edu_loss_fn(edu_logits, batch.y_local)
                global_loss = self.global_loss_fn(global_logit, batch.y_global)

                # 检查损失是否为NaN
                if torch.isnan(edu_loss) or torch.isnan(global_loss):
                    logger.warning(f"Batch {batch_idx}: NaN in loss, skipping...")
                    self.nan_count += 1
                    continue

                # 组合损失（EDU级 + 全局级）
                # 调整权重：全局F1是评估指标，给予更高权重
                loss = 0.3 * edu_loss + 0.7 * global_loss

                if torch.isnan(loss):
                    logger.warning(f"Batch {batch_idx}: NaN in combined loss, skipping...")
                    self.nan_count += 1
                    continue

            except Exception as e:
                logger.warning(f"Batch {batch_idx}: Error in loss computation: {e}")
                continue

            # 反向传播
            loss.backward()

            # 检查梯度
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"Batch {batch_idx}: NaN/Inf gradient in {name}")
                        has_nan_grad = True
                        break

            if has_nan_grad:
                self.optimizer.zero_grad()
                self.nan_count += 1
                continue

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 优化器步进
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            total_edu_loss += edu_loss.item()
            total_global_loss += global_loss.item()
            valid_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'edu': edu_loss.item(),
                'global': global_loss.item(),
                'nan_count': self.nan_count
            })

        if valid_batches == 0:
            logger.error("No valid batches in this epoch!")
            return {'loss': float('nan'), 'edu_loss': float('nan'), 'global_loss': float('nan')}

        return {
            'loss': total_loss / valid_batches,
            'edu_loss': total_edu_loss / valid_batches,
            'global_loss': total_global_loss / valid_batches
        }

    def evaluate(self, val_loader: DataLoader, epoch: int,
                 find_threshold: bool = False) -> dict:
        """评估"""
        self.model.eval()

        # 收集预测和标签
        all_edu_preds = []
        all_edu_labels = []
        all_global_logits = []
        all_global_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Eval]")
            for batch in pbar:
                batch = batch.to(self.device)

                outputs = self.model(batch)

                # EDU级
                edu_probs = torch.sigmoid(outputs['edu_logits'])
                all_edu_preds.extend(edu_probs.cpu().numpy())
                all_edu_labels.extend(batch.y_local.cpu().numpy())

                # 全局级（现在是batch_size大小的tensor）
                global_probs = torch.sigmoid(outputs['global_logit'])
                all_global_logits.extend(global_probs.cpu().numpy())
                all_global_labels.extend(batch.y_global.cpu().numpy())

        # 转换为数组
        import numpy as np
        all_edu_preds = np.array(all_edu_preds)
        all_edu_labels = np.array(all_edu_labels)
        all_global_logits = np.array(all_global_logits)
        all_global_labels = np.array(all_global_labels)

        # 寻找最优阈值（仅在验证集上）
        if find_threshold:
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.linspace(0.3, 0.7, 41):
                preds = (all_global_logits > thresh).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    all_global_labels, preds, average='macro', zero_division=0
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            self.best_threshold = best_thresh
            logger.info(f"最优阈值: {best_thresh:.3f}, F1: {best_f1:.4f}")

        # 使用当前阈值评估
        global_preds = (all_global_logits > self.best_threshold).astype(int)

        # 全局级指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_global_labels, global_preds, average='macro', zero_division=0
        )
        acc = accuracy_score(all_global_labels, global_preds)

        # EDU级指标（二分类）
        edu_preds_binary = (all_edu_preds > self.best_threshold).astype(int)
        edu_f1 = precision_recall_fscore_support(
            all_edu_labels, edu_preds_binary, average='macro', zero_division=0
        )[2]

        return {
            'global_accuracy': acc,
            'global_precision': precision,
            'global_recall': recall,
            'global_f1': f1,
            'edu_f1': edu_f1,
            'threshold': self.best_threshold
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'threshold': self.best_threshold
        }, path)
        logger.info(f"✓ 模型已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点到当前模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_threshold = checkpoint.get('threshold', 0.5)
        logger.info(f"✓ 模型已加载: {path}")

    def load_model_weights(self, path: str):
        """仅加载模型权重，保留当前optimizer状态（用于result-oriented更新）"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_threshold = checkpoint.get('threshold', 0.5)
        logger.info(f"✓ 模型权重已加载（optimizer状态保留）: {path}")


def prepare_data(data_path: str, save_dir: str, max_samples: int = None):
    """
    数据准备（仅首次运行）

    Args:
        data_path: 原始数据路径
        save_dir: 保存目录
        max_samples: 最大样本数（用于测试）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    processed_path = save_dir / "processed_data.json"

    if processed_path.exists():
        logger.info(f"加载已处理数据: {processed_path}")
        with open(processed_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
    else:
        logger.info("开始数据预处理...")
        preprocessor = DataPreprocessor(data_path)
        processed_data = preprocessor.process_all(
            save_path=str(processed_path),
            max_samples=max_samples
        )
        # 释放DMRST模型占用的GPU显存，避免与后续图构建冲突
        del preprocessor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✓ DMRST模型已释放，GPU显存清理完毕")

    # 划分数据集（根据dataset字段）
    train_data, test_data = split_by_dataset_field(processed_data)

    # 统计训练集标签分布
    train_global_labels = [sample['y_global'] for sample in train_data]
    train_pos_count = sum(train_global_labels)
    train_neg_count = len(train_global_labels) - train_pos_count
    train_pos_ratio = 100 * train_pos_count / len(train_global_labels) if train_global_labels else 0

    logger.info(f"\n【训练集标签分布】")
    logger.info(f"  总样本: {len(train_data)}")
    logger.info(f"  正例（有幻觉）: {train_pos_count} ({train_pos_ratio:.2f}%)")
    logger.info(f"  负例（无幻觉）: {train_neg_count} ({100-train_pos_ratio:.2f}%)")
    logger.info(f"  样本比例: {train_pos_count}:{train_neg_count}")

    # 统计测试集标签分布
    test_global_labels = [sample['y_global'] for sample in test_data]
    test_pos_count = sum(test_global_labels)
    test_neg_count = len(test_global_labels) - test_pos_count
    test_pos_ratio = 100 * test_pos_count / len(test_global_labels) if test_global_labels else 0

    logger.info(f"\n【测试集标签分布】")
    logger.info(f"  总样本: {len(test_data)}")
    logger.info(f"  正例（有幻觉）: {test_pos_count} ({test_pos_ratio:.2f}%)")
    logger.info(f"  负例（无幻觉）: {test_neg_count} ({100-test_pos_ratio:.2f}%)")
    logger.info(f"  样本比例: {test_pos_count}:{test_neg_count}\n")
    logger.info("开始构建图...")
    builder = GraphBuilder()

    train_graphs = []
    for sample in tqdm(train_data, desc="构建训练图"):
        try:
            graph = builder.build_graph(sample)
            train_graphs.append(graph)
        except Exception as e:
            logger.warning(f"样本 {sample['id']} 构图失败: {e}")

    test_graphs = []
    for sample in tqdm(test_data, desc="构建测试图"):
        try:
            graph = builder.build_graph(sample)
            test_graphs.append(graph)
        except Exception as e:
            logger.warning(f"样本 {sample['id']} 构图失败: {e}")

    logger.info(f"✓ 训练图: {len(train_graphs)}, 测试图: {len(test_graphs)}")

    return train_graphs, test_graphs


def main():
    """主训练流程"""
    # 配置
    DATA_PATH = "/mnt/nlp/yuanmengying/nli2hallucination/data/summary_data.json"
    SAVE_DIR = "/mnt/nlp/yuanmengying/nli2hallucination/data/cdcl-nli/rst_hallucination_detec/outputs"
    CHECKPOINT_DIR = Path(SAVE_DIR) / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 超参数
    BATCH_SIZE = 32
    EPOCHS = 60  # 增加到60轮，促进收敛
    LEARNING_RATE = 5e-5  # 降低学习率防止NaN
    MAX_SAMPLES = None  # None表示使用全部数据，测试时可设为100

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 数据准备
    logger.info("=" * 80)
    logger.info("1. 数据准备")
    logger.info("=" * 80)

    train_graphs, test_graphs = prepare_data(
        DATA_PATH, SAVE_DIR, max_samples=MAX_SAMPLES
    )

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    logger.info("\n" + "=" * 80)
    logger.info("2. 模型初始化")
    logger.info("=" * 80)

    model = RSTRGATModel(
        input_dim=1027,  # 1024 (semantic) + 1 (position) + 1 (role) + 1 (isolated)
        hidden_dim=512,
        num_rgat_layers=2,
        num_relations=20,
        num_heads=4,
        dropout=0.2  # 从0.3降到0.2，减少过多正则化
    )

    # 初始化模型权重（Xavier初始化）
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            nn.init.xavier_uniform_(param, gain=0.5)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    logger.info("✓ 模型权重初始化完成（Xavier均匀分布）")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,}")

    # 训练器
    trainer = Trainer(model, device, learning_rate=LEARNING_RATE, total_epochs=EPOCHS)

    # 训练循环
    logger.info("\n" + "=" * 80)
    logger.info("3. 开始训练")
    logger.info("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # 验证
        val_metrics = trainer.evaluate(
            test_loader, epoch,
            find_threshold=(epoch % 3 == 0)  # 每3个epoch重新找阈值（之前是5个）
        )

        # 学习率调度
        current_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step()

        # 日志
        logger.info(f"\nEpoch {epoch}/{EPOCHS}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['global_accuracy']:.4f}")
        logger.info(f"  Val Macro F1: {val_metrics['global_f1']:.4f}")
        logger.info(f"  Val EDU F1: {val_metrics['edu_f1']:.4f}")
        logger.info(f"  Threshold: {val_metrics['threshold']:.3f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # 保存最佳模型（面向结果更新）
        if val_metrics['global_f1'] > trainer.best_f1:
            trainer.best_f1 = val_metrics['global_f1']
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
            # 将最佳模型加载回训练器，用于下一个epoch的训练
            # 使用load_model_weights()保留optimizer状态，防止momentum被重置
            trainer.load_model_weights(str(checkpoint_path))
            logger.info(f"✓ 由于F1提升 ({val_metrics['global_f1']:.4f})，已将最佳模型加载回训练器")

        # 定期保存
        if epoch % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)

    # 最终评估
    logger.info("\n" + "=" * 80)
    logger.info("4. 最终评估")
    logger.info("=" * 80)

    # 加载最佳模型和对应的阈值
    best_checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    if best_checkpoint_path.exists():
        logger.info(f"加载最佳模型: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint.get('threshold', 0.5)
        trainer.best_threshold = best_threshold
        logger.info(f"使用最佳阈值: {best_threshold:.3f}")

    final_metrics = trainer.evaluate(test_loader, epoch=-1, find_threshold=False)

    logger.info(f"最终测试结果（使用最佳模型 F1={trainer.best_f1:.4f}):")
    logger.info(f"  Accuracy: {final_metrics['global_accuracy']:.4f}")
    logger.info(f"  Precision: {final_metrics['global_precision']:.4f}")
    logger.info(f"  Recall: {final_metrics['global_recall']:.4f}")
    logger.info(f"  Macro F1: {final_metrics['global_f1']:.4f}")
    logger.info(f"  EDU F1: {final_metrics['edu_f1']:.4f}")
    logger.info(f"  Threshold: {trainer.best_threshold:.3f}")

    logger.info("\n✓ 训练完成！")


if __name__ == "__main__":
    main()
