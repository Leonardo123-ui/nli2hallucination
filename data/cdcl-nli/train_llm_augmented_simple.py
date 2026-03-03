"""
Simplified LLM+CDCL training script for debugging
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from path_ini import data_model_loader
from build_base_graph_extract import ExplainableHeteroClassifier
import dgl

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f"train_llm_augmented_simple_{timestamp}.log")

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

rel_names_long = [
    "Temporal", "TextualOrganization", "Joint", "Topic-Comment", "Comparison",
    "Condition", "Contrast", "Evaluation", "Topic-Change", "Summary", "Manner-Means",
    "Attribution", "Cause", "Background", "Enablement", "Explanation", "Same-Unit",
    "Elaboration", "span", "lexical",
]

def collate_fn(batch):
    (g_premise, g_hypothesis, lexical_chains, nli_label,) = zip(*batch)
    return (list(g_premise), list(g_hypothesis), list(lexical_chains), list(nli_label),)

def main():
    logger.info("=" * 70)
    logger.info("Simplified LLM+CDCL Training Starting")
    logger.info("=" * 70)

    # Load data and configuration
    logger.info("Loading training data and model configuration...")
    try:
        config, train_dataset, test_dataset = data_model_loader(device)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    config.device = device
    config.save_dir = "checkpoints/llm_augmented_simple"
    config.epochs = 2  # Just 2 epochs for testing
    config.lr = 2e-5
    config.batch_size = 4

    # Initialize GNN model
    logger.info("Initializing ExplainableHeteroClassifier...")
    try:
        model = ExplainableHeteroClassifier(
            in_dim=config.model_config["in_dim"],
            hidden_dim=config.model_config["hidden_dim"],
            n_classes=config.model_config["n_classes"],
            rel_names=rel_names_long,
            device=device,
        ).to(device)
        logger.info("[✓] GNN model initialized")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # Create dataloaders
    logger.info("Creating dataloaders...")
    from train import get_dataloader
    train_loader = get_dataloader(train_dataset, config.batch_size, 0)
    test_loader = get_dataloader(test_dataset, config.batch_size, 0, shuffle=False)

    logger.info(f"✓ Train batches: {len(train_loader)}")
    logger.info(f"✓ Test batches: {len(test_loader)}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    num_training_steps = config.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    logger.info("=" * 70)
    logger.info("Starting training loop")
    logger.info("=" * 70)

    # Simple training loop
    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            try:
                graphs1, graphs2, lexical_chains, nli_labels = batch_data
                
                # Remove node_id attributes
                for g in graphs1:
                    if "node_id" in g.ndata:
                        g.ndata.pop("node_id")
                for g in graphs2:
                    if "node_id" in g.ndata:
                        g.ndata.pop("node_id")
                
                # Merge and batch graphs
                combined_graphs = [
                    model.merge_graphs(g_p1, g_p2, lc, rel_names_long)
                    for g_p1, g_p2, lc in zip(graphs1, graphs2, lexical_chains)
                ]
                
                g_batch_combined = dgl.batch(combined_graphs).to(device)
                g_batch1 = dgl.batch(graphs1).to(device)
                g_batch2 = dgl.batch(graphs2).to(device)
                
                # Convert labels
                nli_labels = [label if label == 0 else 1 for label in nli_labels]
                targets = torch.tensor(nli_labels, dtype=torch.long, device=device)
                
                # Forward pass
                graph_repr = model.get_graph_repr(g_batch_combined)
                g_repr1 = model.get_graph_repr(g_batch1)
                g_repr2 = model.get_graph_repr(g_batch2)
                logits = model.classify(g_repr1, g_repr2, graph_repr)
                
                # Loss and backward
                loss = torch.nn.functional.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                train_preds.extend(predictions)
                train_labels.extend(nli_labels)
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx} error: {e}")
                continue
        
        # Log training metrics
        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0) if train_labels else 0.0
        logger.info(f"Train Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}")
        
        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        eval_preds = []
        eval_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Evaluating"):
                try:
                    graphs1, graphs2, lexical_chains, nli_labels = batch_data
                    
                    for g in graphs1:
                        if "node_id" in g.ndata:
                            g.ndata.pop("node_id")
                    for g in graphs2:
                        if "node_id" in g.ndata:
                            g.ndata.pop("node_id")
                    
                    combined_graphs = [
                        model.merge_graphs(g_p1, g_p2, lc, rel_names_long)
                        for g_p1, g_p2, lc in zip(graphs1, graphs2, lexical_chains)
                    ]
                    
                    g_batch_combined = dgl.batch(combined_graphs).to(device)
                    g_batch1 = dgl.batch(graphs1).to(device)
                    g_batch2 = dgl.batch(graphs2).to(device)
                    
                    nli_labels = [label if label == 0 else 1 for label in nli_labels]
                    targets = torch.tensor(nli_labels, dtype=torch.long, device=device)
                    
                    graph_repr = model.get_graph_repr(g_batch_combined)
                    g_repr1 = model.get_graph_repr(g_batch1)
                    g_repr2 = model.get_graph_repr(g_batch2)
                    logits = model.classify(g_repr1, g_repr2, graph_repr)
                    
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    eval_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                    eval_preds.extend(predictions)
                    eval_labels.extend(nli_labels)
                    
                except Exception as e:
                    logger.warning(f"Eval error: {e}")
                    continue
        
        # Log eval metrics
        avg_eval_loss = eval_loss / max(len(test_loader), 1)
        eval_f1 = f1_score(eval_labels, eval_preds, average="macro", zero_division=0) if eval_labels else 0.0
        logger.info(f"Eval Loss: {avg_eval_loss:.4f}, F1: {eval_f1:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("Training completed")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
