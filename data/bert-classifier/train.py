#!/usr/bin/env python3
"""
Hallucination Detector Training Script
Train BERT/RoBERTa models for binary classification of hallucination detection
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import json
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HallucinationDataset(Dataset):
    """Hallucination detection dataset class"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Encode text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """Load training data"""
    logger.info(f"Loading data from: {data_path}")
    
    # Load data from Excel file
    train_df = pd.read_excel(data_path, sheet_name='训练集')
    test_df = pd.read_excel(data_path, sheet_name='测试集')
    
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    return train_df, test_df

def prepare_texts(df, input_format='context_output'):
    """
    Prepare input text
    input_format options:
    - 'context_output': Use [CLS] context [SEP] output [SEP]
    - 'context_only': Only use context
    - 'output_only': Only use output
    """
    if input_format == 'context_output':
        texts = [f"{row['context']} [SEP] {row['output']}" for _, row in df.iterrows()]
    elif input_format == 'context_only':
        texts = df['context'].tolist()
    elif input_format == 'output_only':
        texts = df['output'].tolist()
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    return texts

def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False negative rate (miss rate)
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # False positive rate (false alarm rate)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

def train_model(
    model_name='bert-base-uncased',
    data_path='../summary_nli_hallucination_dataset.xlsx',
    output_dir='./models',
    input_format='context_output',
    max_length=512,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    save_steps=500,
    eval_steps=500,
    warmup_steps=500,
    weight_decay=0.01
):
    """Train hallucination detection model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_df, test_df = load_data(data_path)
    
    # Prepare texts
    train_texts = prepare_texts(train_df, input_format)
    train_labels = train_df['label'].tolist()
    
    test_texts = prepare_texts(test_df, input_format)
    test_labels = test_df['label'].tolist()
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "no_hallucination", 1: "hallucination"},
        label2id={"no_hallucination": 0, "hallucination": 1}
    )
    
    # Create datasets
    train_dataset = HallucinationDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = HallucinationDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None  # Disable wandb reporting
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    final_model_path = f"{output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'input_format': input_format,
        'max_length': max_length,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'final_eval_results': eval_results,
        'training_time': datetime.now().isoformat(),
        'model_path': final_model_path
    }
    
    with open(f"{output_dir}/training_info.json", 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Final evaluation results: {eval_results}")
    
    return trainer, eval_results

def main():
    parser = argparse.ArgumentParser(description='Train hallucination detection model')
    parser.add_argument('--model_name', default='bert-base-uncased', 
                       help='Pre-trained model name (default: bert-base-uncased)')
    parser.add_argument('--data_path', default='../summary_nli_hallucination_dataset.xlsx',
                       help='Dataset path')
    parser.add_argument('--output_dir', default='./models/hallucination_detector',
                       help='Model save directory')
    parser.add_argument('--input_format', default='context_output',
                       choices=['context_output', 'context_only', 'output_only'],
                       help='Input text format')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps')
    
    args = parser.parse_args()
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Start training
    trainer, eval_results = train_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        input_format=args.input_format,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    print("\n" + "="*50)
    print("Training completed! Main evaluation metrics:")
    print("="*50)
    print(f"Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"Recall: {eval_results.get('eval_recall', 0):.4f}")
    print(f"F1 Score: {eval_results.get('eval_f1', 0):.4f}")
    print(f"Specificity: {eval_results.get('eval_specificity', 0):.4f}")
    print(f"Miss Rate (FNR): {eval_results.get('eval_false_negative_rate', 0):.4f}")
    print(f"False Alarm Rate (FPR): {eval_results.get('eval_false_positive_rate', 0):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()