"""
LLM+CDCL 增强型训练脚本
集成Qwen LLM和本地判别模型的联合训练
直接使用train.py中的ExplainableHeteroClassifier
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from main import QwenLLMClient, QwenConfig, ModelDeployType
from path_ini import data_model_loader
from build_base_graph_extract import ExplainableHeteroClassifier
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import numpy as np
import dgl
import json
import hashlib
import base64

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_model_llm1.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 关系名称列表（从train.py复制）
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
    "lexical",
]


def extract_text_from_graph(graph: dgl.DGLGraph) -> str:
    """
    从DGL图中提取完整的文本内容

    Args:
        graph: DGL图对象

    Returns:
        拼接后的完整文本字符串
    """
    try:
        if "text_encoded" not in graph.ndata:
            logger.warning("图中没有text_encoded属性")
            return ""

        texts = []
        num_nodes = graph.num_nodes()

        for node_id in range(num_nodes):
            encoded = graph.ndata["text_encoded"][node_id]

            # 转换为字节
            if isinstance(encoded, torch.Tensor):
                byte_data = bytes(encoded.cpu().numpy().astype(int))
            else:
                byte_data = bytes(encoded)

            # Base64解码
            try:
                byte_data = byte_data.rstrip(b"\0")
                decoded_text = base64.b64decode(byte_data).decode("utf-8")
                if decoded_text.strip():  # 过滤空文本
                    texts.append(decoded_text.strip())
            except Exception as e:
                logger.debug(f"解码节点{node_id}文本失败: {e}")
                continue

        # 拼接所有节点文本
        full_text = " ".join(texts)
        return full_text

    except Exception as e:
        logger.warning(f"提取图文本失败: {e}")
        return ""


def compute_sample_hash(premise_text: str, hypothesis_text: str) -> str:
    """
    计算样本的唯一标识哈希值，用于缓存

    Args:
        premise_text: 前提文本
        hypothesis_text: 假设文本

    Returns:
        MD5哈希字符串
    """
    content = f"{premise_text}|||{hypothesis_text}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


class RationaleCache:
    """Rationale缓存管理器"""

    def __init__(self, cache_dir: str = "cache/rationales"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "rationale_cache.json"

        # 加载现有缓存
        self.cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"加载了 {len(self.cache)} 条缓存的rationale")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.cache = {}

    def get(self, sample_hash: str) -> str:
        """获取缓存的rationale"""
        return self.cache.get(sample_hash, None)

    def set(self, sample_hash: str, rationale: str):
        """保存rationale到缓存"""
        self.cache[sample_hash] = rationale

    def save(self):
        """将缓存持久化到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"缓存已保存，共 {len(self.cache)} 条记录")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def __len__(self):
        return len(self.cache)



class LLMAugmentedTrainer:
    """LLM增强型训练器"""

    def __init__(self, model, llm_config=None, use_cache=True):
        """
        初始化训练器

        Args:
            model: ExplainableHeteroClassifier 模型实例
            llm_config: LLM配置对象
            use_cache: 是否使用rationale缓存
        """
        logger.info(f"初始化LLM+CDCL训练器...")
        logger.info(f"设备: {device}")

        # 初始化LLM客户端
        if llm_config is None:
            llm_config = QwenConfig(
                model_name="/mnt/second/yuanmengying/qwen3-8b",
                deploy_type=ModelDeployType.HUGGINGFACE,
            )
        self.llm_client = QwenLLMClient(config=llm_config)
        logger.info("[✓] LLM客户端初始化完成")

        # 保存模型
        self.model = model
        logger.info("[✓] ExplainableHeteroClassifier模型已加载")

        # 初始化缓存
        self.use_cache = use_cache
        if use_cache:
            self.rationale_cache = RationaleCache()
            logger.info(f"[✓] Rationale缓存已启用，已有 {len(self.rationale_cache)} 条缓存")
        else:
            self.rationale_cache = None
            logger.info("[✓] Rationale缓存已禁用")

    def get_rationale(self, premise_text: str, hypothesis_text: str) -> str:
        """
        获取样本的rationale（优先从缓存，缓存未命中则调用LLM）

        Args:
            premise_text: 前提文本
            hypothesis_text: 假设文本

        Returns:
            rationale文本
        """
        # 计算样本哈希
        sample_hash = compute_sample_hash(premise_text, hypothesis_text)

        # 检查缓存
        if self.use_cache and self.rationale_cache is not None:
            cached_rationale = self.rationale_cache.get(sample_hash)
            if cached_rationale is not None:
                return cached_rationale

        # 缓存未命中，调用LLM
        try:
            rationale = self.llm_client.call_llm(
                system_prompt=(
                    "你是一名自然语言推理(NLI)分析专家。请分析给定的前提(Premise)和假设(Hypothesis)之间的逻辑关系，"
                    "重点关注：1) 语义矛盾之处 2) 关键信息缺失 3) 逻辑推理链条。"
                ),
                user_input=(
                    f"前提: {premise_text}\n"
                    f"假设: {hypothesis_text}\n\n"
                    f"请简要分析它们之间的关键语义关系和潜在不一致之处（50-100字）。"
                ),
                max_tokens=150
            )

            # 保存到缓存
            if self.use_cache and self.rationale_cache is not None:
                self.rationale_cache.set(sample_hash, rationale)

            return rationale

        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
            return ""

    def generate_batch_rationales(self, graphs1, graphs2):
        """
        为一个batch的样本生成rationales（带缓存机制）

        Args:
            graphs1: premise图列表
            graphs2: hypothesis图列表

        Returns:
            rationales列表
        """
        rationales = []
        llm_calls = 0
        cache_hits = 0
        text_extraction_failures = 0

        for sample_idx in range(len(graphs1)):
            try:
                # 提取实际文本
                premise_text = extract_text_from_graph(graphs1[sample_idx])
                hypothesis_text = extract_text_from_graph(graphs2[sample_idx])

                if not premise_text or not hypothesis_text:
                    logger.debug(f"样本{sample_idx}文本提取失败: premise_len={len(premise_text)}, hyp_len={len(hypothesis_text)}")
                    text_extraction_failures += 1
                    rationales.append("")
                    continue

                # 计算哈希并检查缓存
                sample_hash = compute_sample_hash(premise_text, hypothesis_text)

                if self.use_cache and self.rationale_cache is not None:
                    cached = self.rationale_cache.get(sample_hash)
                    if cached is not None:
                        rationales.append(cached)
                        cache_hits += 1
                        continue

                # 调用LLM
                rationale = self.get_rationale(premise_text, hypothesis_text)
                rationales.append(rationale)
                llm_calls += 1

            except Exception as e:
                logger.warning(f"为样本{sample_idx}生成rationale失败: {e}")
                text_extraction_failures += 1
                rationales.append("")

        if text_extraction_failures > 0:
            logger.warning(f"Batch统计: 文本提取失败={text_extraction_failures}/{len(graphs1)}")

        logger.debug(f"Batch rationale统计: LLM调用={llm_calls}, 缓存命中={cache_hits}, 失败={text_extraction_failures}")
        return rationales



    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # 提取批次数据
                if len(batch_data) == 4:
                    graphs1, graphs2, lexical_chains, nli_labels = batch_data

                    # ========== 第一步：为batch生成LLM Rationales（带缓存）==========
                    rationales = self.generate_batch_rationales(graphs1, graphs2)

                    # ========== 第二步：应用LLM增强到图 ==========
                    from llm_graph_augmentation import augment_batch_graphs
                    try:
                        graphs1, graphs2 = augment_batch_graphs(
                            graphs1, graphs2, rationales, method="multiply"
                        )
                    except Exception as e:
                        logger.debug(f"LLM增强失败: {e}，继续使用原始图")

                    # ========== 第三步：批处理图并移到设备 ==========
                    for g in graphs1:
                        if "node_id" in g.ndata:
                            g.ndata.pop("node_id")
                    for g in graphs2:
                        if "node_id" in g.ndata:
                            g.ndata.pop("node_id")

                    # 合并图
                    combined_graphs = [
                        self.model.merge_graphs(g_p1, g_p2, lc, rel_names_long)
                        for g_p1, g_p2, lc in zip(graphs1, graphs2, lexical_chains)
                    ]

                    g_batch_combined = dgl.batch(combined_graphs).to(device)
                    g_batch1 = dgl.batch(graphs1).to(device)
                    g_batch2 = dgl.batch(graphs2).to(device)

                    # 转换标签
                    nli_labels = [label if label == 0 else 1 for label in nli_labels]
                    targets = torch.tensor(nli_labels, dtype=torch.long, device=device)

                    # ========== 第四步：模型前向传播 ==========
                    graph_repr = self.model.get_graph_repr(g_batch_combined)
                    g_repr1 = self.model.get_graph_repr(g_batch1)
                    g_repr2 = self.model.get_graph_repr(g_batch2)

                    logits = self.model.classify(g_repr1, g_repr2, graph_repr)

                    # ========== 第五步：计算损失 ==========
                    from torch.nn import CrossEntropyLoss
                    criterion = CrossEntropyLoss()
                    batch_loss = criterion(logits, targets)

                    # ========== 第六步：反向传播和优化 ==========
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    # ========== 第七步：记录指标 ==========
                    epoch_loss += batch_loss.item()
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                    all_predictions.extend(predictions)
                    all_labels.extend(nli_labels)

                    # 计算当前batch的F1
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    batch_f1 = f1_score(nli_labels, batch_preds, average="macro", zero_division=0)

                    progress_bar.set_postfix({
                        "loss": batch_loss.item(),
                        "f1": batch_f1
                    })
                else:
                    continue

            except Exception as e:
                logger.warning(f"批次 {batch_idx} 处理出错: {e}")
                continue

        # 计算epoch指标
        epoch_metrics = {
            "loss": epoch_loss / max(len(train_loader), 1),
            "predictions": all_predictions,
            "labels": all_labels
        }

        if all_labels and all_predictions:
            epoch_metrics["accuracy"] = accuracy_score(all_labels, all_predictions)
            epoch_metrics["f1"] = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        # 保存缓存
        if self.use_cache and self.rationale_cache is not None:
            self.rationale_cache.save()

        return epoch_metrics

    def eval_epoch(self, eval_loader, epoch):
        """评估一个epoch"""
        if self.model is None:
            logger.warning("演示模式：跳过实际评估")
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
            }

        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(eval_loader, desc=f"Epoch {epoch} Evaluation")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    if len(batch_data) == 4:
                        graphs1, graphs2, lexical_chains, nli_labels = batch_data

                        # ========== 评估时也使用LLM增强（确保一致性）==========
                        rationales = self.generate_batch_rationales(graphs1, graphs2)

                        # 应用LLM增强
                        from llm_graph_augmentation import augment_batch_graphs
                        try:
                            graphs1, graphs2 = augment_batch_graphs(
                                graphs1, graphs2, rationales, method="multiply"
                            )
                        except Exception as e:
                            logger.debug(f"评估时LLM增强失败: {e}，继续使用原始图")

                        import dgl

                        # 移除node_id属性
                        for g in graphs1:
                            if "node_id" in g.ndata:
                                g.ndata.pop("node_id")
                        for g in graphs2:
                            if "node_id" in g.ndata:
                                g.ndata.pop("node_id")

                        # 批处理
                        device = next(self.model.parameters()).device
                        g_batch1 = dgl.batch(graphs1).to(device)
                        g_batch2 = dgl.batch(graphs2).to(device)

                        # 合并图
                        combined_graphs = [
                            self.model.merge_graphs(g_p1, g_p2, lc, rel_names_long)
                            for g_p1, g_p2, lc in zip(graphs1, graphs2, lexical_chains)
                        ]
                        g_batch_combined = dgl.batch(combined_graphs).to(device)

                        # 转换标签
                        nli_labels = [label if label == 0 else 1 for label in nli_labels]
                        targets = torch.tensor(nli_labels, dtype=torch.long, device=device)

                        # 前向传播
                        graph_repr = self.model.get_graph_repr(g_batch_combined)
                        g_repr1 = self.model.get_graph_repr(g_batch1)
                        g_repr2 = self.model.get_graph_repr(g_batch2)
                        logits = self.model.classify(g_repr1, g_repr2, graph_repr)

                        # 计算损失
                        from torch.nn import CrossEntropyLoss
                        criterion = CrossEntropyLoss()
                        batch_loss = criterion(logits, targets)
                        epoch_loss += batch_loss.item()

                        # 收集预测
                        predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                        all_predictions.extend(predictions)
                        all_labels.extend(nli_labels)

                        # 计算当前batch的F1
                        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        batch_f1 = f1_score(nli_labels, batch_preds, average="macro", zero_division=0)

                        progress_bar.set_postfix({
                            "loss": batch_loss.item(),
                            "f1": batch_f1
                        })
                    else:
                        continue

                except Exception as e:
                    logger.warning(f"评估批次 {batch_idx} 出错: {e}", exc_info=True)
                    continue

        # 计算指标
        metrics = {
            "loss": epoch_loss / max(len(eval_loader), 1),
        }

        if all_labels and all_predictions:
            metrics["accuracy"] = accuracy_score(all_labels, all_predictions)
            metrics["f1"] = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
            metrics["precision"] = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
            metrics["recall"] = recall_score(all_labels, all_predictions, average="macro", zero_division=0)

        # 保存缓存
        if self.use_cache and self.rationale_cache is not None:
            self.rationale_cache.save()

        return metrics

    def save_model(self, path, optimizer=None, epoch=None):
        """保存模型"""
        if self.model is None:
            logger.warning("演示模式：无模型可保存")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
        if epoch:
            save_dict["epoch"] = epoch

        torch.save(save_dict, path)
        logger.info(f"模型已保存到: {path}")


def main():
    """主训练函数"""
    logger.info("=" * 70)
    logger.info("LLM+CDCL 增强型训练开始")
    logger.info("=" * 70)

    # 加载数据和配置
    logger.info("加载训练数据和模型配置...")
    try:
        config, train_dataset, test_dataset = data_model_loader(device)
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        logger.info("进入演示模式")
        config = None
        train_dataset = None
        test_dataset = None

    # 配置参数
    if config:
        config.device = device
        config.save_dir = "checkpoints/llm_augmented_model"
        config.epochs = 50  # 完整训练50个epoch
        config.lr = 2e-5
        config.batch_size = 30
    else:
        # 创建虚拟配置用于演示
        class DummyConfig:
            device = device
            save_dir = "checkpoints/llm_augmented_model"
            epochs = 3
            lr = 2e-5
            batch_size = 2
        config = DummyConfig()

    # 初始化GNN模型（ExplainableHeteroClassifier）
    logger.info("初始化ExplainableHeteroClassifier模型...")
    try:
        model = ExplainableHeteroClassifier(
            in_dim=config.model_config["in_dim"],
            hidden_dim=config.model_config["hidden_dim"],
            n_classes=config.model_config["n_classes"],
            rel_names=rel_names_long,
            device=device,
        ).to(device)
        logger.info("[✓] GNN模型初始化完成")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        logger.info("进入演示模式")
        model = None

    # 初始化训练器
    if model is not None:
        trainer = LLMAugmentedTrainer(
            model=model,
            llm_config=QwenConfig(
                model_name="/mnt/second/yuanmengying/qwen3-8b",
                deploy_type=ModelDeployType.HUGGINGFACE,
            )
        )
    else:
        trainer = None

    # 如果训练器初始化成功且有数据，设置优化器和调度器
    if trainer is not None and train_dataset is not None:
        # 创建数据加载器
        logger.info("创建数据加载器...")
        from train import get_dataloader
        train_loader = get_dataloader(train_dataset, config.batch_size, 0)
        test_loader = get_dataloader(test_dataset, config.batch_size, 0, shuffle=False)

        logger.info(f"✓ 训练批次数: {len(train_loader)}")
        logger.info(f"✓ 验证批次数: {len(test_loader)}")

        # 创建优化器和调度器
        logger.info("创建优化器和调度器...")
        optimizer = AdamW(
            trainer.model.parameters(),
            lr=config.lr,
            weight_decay=1e-4,
        )

        num_training_steps = config.epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps,
        )

        logger.info(f"✓ 总训练步数: {num_training_steps}")

        # 训练循环
        logger.info("\n" + "=" * 70)
        logger.info("开始LLM+CDCL增强型训练")
        logger.info("=" * 70)

        best_f1 = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(config.epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"[Epoch {epoch+1}/{config.epochs}]")
            logger.info(f"{'='*70}")

            # 训练阶段
            train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
            logger.info(f"✓ 训练完成:")
            logger.info(f"  - 损失: {train_metrics['loss']:.4f}")
            if 'accuracy' in train_metrics:
                logger.info(f"  - 准确率: {train_metrics['accuracy']:.4f}")
            if 'f1' in train_metrics:
                logger.info(f"  - F1分数: {train_metrics['f1']:.4f}")

            # 评估阶段
            eval_metrics = trainer.eval_epoch(test_loader, epoch)
            logger.info(f"✓ 评估完成:")
            for key, val in eval_metrics.items():
                if isinstance(val, float):
                    logger.info(f"  - {key}: {val:.4f}")

            # 模型保存和早停
            if "f1" in eval_metrics:
                val_f1 = eval_metrics["f1"]
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    model_path = os.path.join(config.save_dir, f"best_llm_augmented_model.pt")
                    trainer.save_model(model_path, optimizer, epoch)
                    logger.info(f"✓✓ 验证F1提升到 {val_f1:.4f}，模型已保存!")
                else:
                    patience_counter += 1
                    logger.info(f"✗ 验证F1未提升 (最佳: {best_f1:.4f}, 当前: {val_f1:.4f}, 耐心: {patience_counter}/{patience})")

                    if patience_counter >= patience:
                        logger.info(f"\n⚠️ 早停触发：{patience}个epoch内F1无提升")
                        logger.info(f"最终最佳F1分数: {best_f1:.4f}")
                        break
    else:
        logger.warning("演示模式: 跳过实际训练")
        logger.info("展示流水线结构...")

        # 演示流水线
        logger.info("\n" + "=" * 70)
        logger.info("LLM+CDCL 流水线演示")
        logger.info("=" * 70)

        sample_text = "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行人类智能任务的机器。"
        logger.info(f"\n输入文本: {sample_text[:50]}...")

        # 调用LLM
        logger.info("\n[阶段 1] LLM理由生成...")
        rationale = trainer.llm_client.call_llm(
            system_prompt="分析文本的关键要点",
            user_input=sample_text,
            max_tokens=50
        )
        logger.info(f"LLM理由: {rationale[:100]}...")

        logger.info("\n[阶段 2] 输入格式化...")
        logger.info(f"格式: [LLM理由] + [SEP] + [摘要] + [SEP] + [原文]")

        logger.info("\n[阶段 3] 本地模型推理...")
        logger.info("准备好进行分类预测")

    logger.info("\n" + "=" * 70)
    logger.info("训练完成")
    logger.info("=" * 70)
    logger.info(f"日志保存到: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {e}", exc_info=True)
