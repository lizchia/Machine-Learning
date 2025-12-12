"""
Script: train_pubmed_bert_trainer.py
Description: Fine-tune PubMedBERT using Hugging Face Trainer API + SHAP analysis.
Refactored based on Team Lead's latest architecture.
"""
from __future__ import annotations

import os
import sys

# GPU device selection - can be overridden by --gpu_device argument
# If CUDA_VISIBLE_DEVICES is set, it will be used, otherwise default to GPU 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Default to GPU 0

import argparse
import random
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, auc, classification_report
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)

# Default Settings
DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# ==========================================
# Core Functions
# ==========================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(csv_path: str, text_col: str, label_col: str, threshold: float = None) -> pd.DataFrame:
    """
    Loads data and handles label generation if needed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到檔案: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 1. Check Text Column
    if text_col not in df.columns:
        raise ValueError(f"❌ 缺少文字欄位: '{text_col}'. 現有欄位: {list(df.columns)}")
    df[text_col] = df[text_col].fillna("No Report").astype(str)

    # 2. Handle Label Logic
    if label_col in df.columns:
        # Check if we need to binarize (if column is continuous like 'los')
        if threshold is not None and df[label_col].dtype in [float, int] and df[label_col].max() > 1:
            print(f"ℹ️  應用閾值 > {threshold} 於欄位 '{label_col}' 產生標籤。")
            df['label'] = (df[label_col] > threshold).astype(int)
        else:
            print(f"ℹ️  直接使用欄位 '{label_col}' 作為標籤。")
            df['label'] = df[label_col].astype(int)
    else:
        # Fallback: Check if user meant to generate from 'los' but passed a different name
        raise ValueError(f"❌ 缺少標籤欄位: '{label_col}'. 現有欄位: {list(df.columns)}")

    df = df.dropna(subset=['label'])
    unique, counts = np.unique(df['label'], return_counts=True)
    print(f"   類別分佈: {dict(zip(unique, counts))}")
    
    return df

class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length" 
        )
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    """
    Calculates detailed metrics: Specificity, NPV, PPV, etc.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle edge case where CM might not be 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0 # Fallback
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) # PPV
    recall = recall_score(y_true, y_pred, zero_division=0)       # Sensitivity
    
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Specificity = TN / (TN + FP)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # NPV = TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5
        auprc = 0.0
        
    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1,
        "Positive F1": f1_pos,
        "Negative F1": f1_neg,
        "NPV": npv,
        "PPV": precision,
        "Specificity": sp,
        "AUROC": auroc,
        "AUPRC": auprc,
        "CM": cm
    }

def plot_save_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - Final Test Set')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 ROC 曲線已儲存: {save_path}")

# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT with Trainer API & SHAP.")
    
    # Path Arguments
    parser.add_argument("--csv_path", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument("--output_dir", type=str, default="./pubmed_results", help="Root output directory.")
    
    # Data Arguments
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="target_long_stay", help="Target column name.")
    parser.add_argument("--threshold", type=float, default=None, help="If using 'los', set threshold (e.g., 7.0).")
    
    # Model Arguments
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_device", type=str, default="CUDA_VISIBLE_DEVICES", help="GPU ID (e.g., '0' or '1').")
    
    args = parser.parse_args()
    
    # 1. Environment Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device} (GPU: {args.gpu_device})")
    
    set_seed(args.seed)
    
    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📂 輸出目錄: {exp_dir}")

    # 2. Load & Split Data
    print("\n📦 Loading and Splitting Data...")
    df = load_data(args.csv_path, args.text_column, args.label_column, args.threshold)
    
    X = df[args.text_column].values
    y = df['label'].values
    
    # Hold-out Test Split (20%) - Matches Team Lead's Logic
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    
    print(f"   Train/Val Set: {len(X_train_full)}")
    print(f"   Test Set:      {len(X_test)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 3. 10-Fold Cross-Validation (on Training Set)
    print(f"\n🔄 Starting {args.folds}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")
        
        # Prepare Fold Data
        train_ds = ClinicalDataset(X_train_full[train_idx], y_train_full[train_idx], tokenizer)
        val_ds = ClinicalDataset(X_train_full[val_idx], y_train_full[val_idx], tokenizer)
        
        # Init Model
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
        
        # Trainer Args
        fold_output_dir = os.path.join(exp_dir, f"fold_{fold+1}")
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            learning_rate=args.lr,
            weight_decay=0.01,
            report_to="none",
            disable_tqdm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer)
        )
        
        # Train
        trainer.train()
        
        # Predict
        preds = trainer.predict(val_ds)
        pred_labels = np.argmax(preds.predictions, axis=1)
        pred_probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1).numpy()[:, 1]
        
        # Metrics
        metrics = calculate_comprehensive_metrics(y_train_full[val_idx], pred_labels, pred_probs)
        fold_results.append(metrics)
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        
        print(f"   Fold {fold+1} Macro F1: {metrics['Macro F1']:.4f}")

    # 4. Save CV Results
    metrics_df = pd.DataFrame(fold_results).drop(columns=['CM']) # Drop CM for CSV
    metrics_df.to_csv(os.path.join(exp_dir, "cv_metrics.csv"), index_label="Fold")
    print("\n📊 CV Results Summary (Mean):")
    print(metrics_df.mean())

    # ==========================================
    # 5. Final Retraining & Evaluation
    # ==========================================
    print(f"\n{'='*40}")
    print("🏆 Training Final Model on Full Train Set")
    print(f"{'='*40}")
    
    final_train_ds = ClinicalDataset(X_train_full, y_train_full, tokenizer)
    final_test_ds = ClinicalDataset(X_test, y_test, tokenizer)
    
    final_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    
    final_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, "final_model"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch", # Save final model
        learning_rate=args.lr,
        report_to="none"
    )
    
    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=final_train_ds
    )
    
    final_trainer.train()
    
    # Save Model
    final_trainer.save_model()
    tokenizer.save_pretrained(os.path.join(exp_dir, "final_model"))
    
    # Final Evaluation
    print("\n📝 Evaluating on Hold-out Test Set...")
    test_res = final_trainer.predict(final_test_ds)
    test_labels = np.argmax(test_res.predictions, axis=1)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_res.predictions), dim=-1).numpy()[:, 1]
    
    final_metrics = calculate_comprehensive_metrics(y_test, test_labels, test_probs)
    
    # Save Final Report
    report_path = os.path.join(exp_dir, "final_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Final Evaluation on Hold-out Test Set\n")
        f.write("="*40 + "\n")
        for k, v in final_metrics.items():
            if k != 'CM':
                f.write(f"{k:<20}: {v:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(final_metrics['CM']))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, test_labels))
        
    print("\nFinal Test Metrics:")
    for k, v in final_metrics.items():
        if k != 'CM': print(f"{k:<20}: {v:.4f}")
        
    # Plot ROC
    plot_save_roc_curve(y_test, test_probs, os.path.join(exp_dir, "final_roc_curve.png"))

    # ==========================================
    # 6. SHAP Analysis
    # ==========================================
    print("\n🧠 Generating SHAP Analysis (Top 5 samples)...")
    try:
        # Create Pipeline
        pred_pipeline = pipeline(
            "text-classification",
            model=final_model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True,
            truncation=True,
            max_length=512
        )
        
        explainer = shap.Explainer(pred_pipeline)
        # Only take 5 samples to avoid OOM / Time issues
        sample_texts = [str(t)[:2000] for t in X_test[:5]] 
        shap_values = explainer(sample_texts)
        
        # Save SHAP as HTML
        shap_html_path = os.path.join(exp_dir, "shap_analysis.html")
        with open(shap_html_path, 'w', encoding='utf-8') as f:
            f.write(shap.plots.text(shap_values, display=False))
        print(f"✅ SHAP 分析已儲存: {shap_html_path}")
        
    except Exception as e:
        print(f"❌ SHAP 分析失敗 (可能是記憶體不足): {e}")

    print(f"\n✨ 全部完成！結果已儲存於: {exp_dir}")

if __name__ == "__main__":
    main()