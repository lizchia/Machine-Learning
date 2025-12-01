"""
Utility script to fine-tune PubMedBERT on a LOS classification dataset.

The logic mirrors the data handling found in `1_train.ipynb` but moves it into
a reusable Python module that can be triggered from the CLI.
"""
from __future__ import annotations

import os
import sys

# GPU device selection - can be overridden by --gpu_device argument
# If CUDA_VISIBLE_DEVICES is set, it will be used, otherwise default to GPU 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Default to GPU 0

import argparse
import json
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Device will be set explicitly in main() based on --gpu_device argument
DEVICE = None  # Will be set in main() before use


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str, text_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset and ensure required columns exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist")

    df = pd.read_csv(csv_path)
    missing_cols = {text_col, label_col} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return np.array(texts), np.array(labels)


class MedicalReportDataset(Dataset):
    """
    Dataset wrapper that tokenizes on the fly with the PubMedBERT tokenizer.
    """

    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@dataclass
class FoldResult:
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float


def train_single_fold(
    model_name: str,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> Tuple[FoldResult, List[int], List[int]]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(batch["labels"].tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0
    )
    fold_result = FoldResult(
        accuracy=accuracy_score(trues, preds),
        f1_macro=f1,
        precision_macro=precision,
        recall_macro=recall,
    )

    del model
    torch.cuda.empty_cache()

    return fold_result, trues, preds


def train_final_model(
    model_name: str,
    texts: np.ndarray,
    labels: np.ndarray,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    save_path: str,
) -> None:
    """
    Train a final model on the full dataset and save it for inference.
    """
    print(f"\n{'='*60}")
    print("Training final model on full dataset for inference...")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Create dataset and dataloader for full dataset
    full_ds = MedicalReportDataset(texts, labels, tokenizer, max_len)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in full_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    # Save model and tokenizer
    model_save_path = os.path.join(save_path, "final_model")
    os.makedirs(model_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"\n✓ Final model saved to: {model_save_path}")
    print(f"  - Model: {os.path.join(model_save_path, 'pytorch_model.bin')}")
    print(f"  - Tokenizer: {os.path.join(model_save_path, 'tokenizer_config.json')}")
    
    del model
    torch.cuda.empty_cache()


def run_cross_validation(
    args: argparse.Namespace, texts: np.ndarray, labels: np.ndarray
) -> Tuple[Dict[str, float], str]:
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        exp_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration
    experiment_config = {
        "experiment_name": args.experiment_name or f"experiment_{timestamp}",
        "timestamp": timestamp,
        "model_name": args.model_name,
        "dataset_path": args.csv_path,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "n_folds": args.folds,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "dataset_size": len(texts),
        "num_classes": len(np.unique(labels)),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    
    with open(os.path.join(exp_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_config['experiment_name']}")
    print(f"Output directory: {exp_dir}")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    metrics: List[FoldResult] = []
    all_trues: List[int] = []
    all_preds: List[int] = []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"\n--- Fold {fold_idx}/{args.folds} ---")
        train_ds = MedicalReportDataset(texts[train_idx], labels[train_idx], tokenizer, args.max_len)
        val_ds = MedicalReportDataset(texts[val_idx], labels[val_idx], tokenizer, args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        fold_result, trues, preds = train_single_fold(
            args.model_name, tokenizer, train_loader, val_loader, args.epochs, args.lr
        )
        metrics.append(fold_result)
        all_trues.extend(trues)
        all_preds.extend(preds)
        
        # Save per-fold results
        fold_detail = {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy": float(fold_result.accuracy),
            "f1_macro": float(fold_result.f1_macro),
            "precision_macro": float(fold_result.precision_macro),
            "recall_macro": float(fold_result.recall_macro),
        }
        fold_details.append(fold_detail)

        print(
            f"Fold {fold_idx} -- Acc: {fold_result.accuracy:.4f}, "
            f"F1(macro): {fold_result.f1_macro:.4f}"
        )

    averaged = {
        "accuracy": float(np.mean([m.accuracy for m in metrics])),
        "accuracy_std": float(np.std([m.accuracy for m in metrics])),
        "f1_macro": float(np.mean([m.f1_macro for m in metrics])),
        "f1_macro_std": float(np.std([m.f1_macro for m in metrics])),
        "precision_macro": float(np.mean([m.precision_macro for m in metrics])),
        "precision_macro_std": float(np.std([m.precision_macro for m in metrics])),
        "recall_macro": float(np.mean([m.recall_macro for m in metrics])),
        "recall_macro_std": float(np.std([m.recall_macro for m in metrics])),
    }

    # Get detailed classification report
    report_text = classification_report(all_trues, all_preds, output_dict=False, digits=4)
    report_dict = classification_report(all_trues, all_preds, output_dict=True, digits=4)
    
    # Save all results
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    
    with open(os.path.join(exp_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2)
    
    with open(os.path.join(exp_dir, "fold_details.json"), "w", encoding="utf-8") as f:
        json.dump(fold_details, f, indent=2)
    
    # Save complete experiment summary
    experiment_summary = {
        **experiment_config,
        "averaged_metrics": averaged,
        "per_fold_results": fold_details,
    }
    
    with open(os.path.join(exp_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)

    print("\n======== Aggregated Metrics ========")
    for key, value in averaged.items():
        if "std" not in key:
            std_key = f"{key}_std"
            std_value = averaged.get(std_key, 0)
            print(f"{key}: {value:.4f} ± {std_value:.4f}")
    print(f"\nResults saved to: {exp_dir}")
    print("\nClassification report:")
    print(report_text)

    return averaged, exp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune PubMedBERT on LOS data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Selection:
  To use a specific GPU, you can either:
  1. Set CUDA_VISIBLE_DEVICES environment variable before running:
     CUDA_VISIBLE_DEVICES=1 python train_pubmed_bert.py ...
  2. Use --gpu argument (note: must be set before torch import, so set env var instead)
  
  GPU indices: 0, 1, 2, etc. (not "GPU 0" or "GPU 1")
        """
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV dataset.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column.")
    parser.add_argument(
        "--label_column", type=str, default="los_category", help="Name of the label column."
    )
    parser.add_argument("--output_dir", type=str, default="./pubmed_outputs", help="Where to save reports.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for this experiment (optional).")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="HF model identifier.")
    parser.add_argument("--max_len", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per fold.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--folds", type=int, default=5, help="StratifiedKFold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_final_model", action="store_true", help="Train and save a final model on full dataset for inference.")
    parser.add_argument("--gpu_device", type=int, default=None, help="Explicit GPU device index to use (e.g., 0, 1, 2). If not specified, uses first available GPU or CPU.")
    return parser.parse_args()


def main() -> None:
    global DEVICE
    args = parse_args()
    
    # Set device explicitly based on user's GPU selection
    if args.gpu_device is not None:
        # User explicitly specified a GPU device
        if torch.cuda.is_available():
            if args.gpu_device >= torch.cuda.device_count():
                print(f"Warning: GPU {args.gpu_device} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
                print("Falling back to CPU.")
                DEVICE = torch.device("cpu")
            else:
                DEVICE = torch.device(f"cuda:{args.gpu_device}")
                torch.cuda.set_device(args.gpu_device)
                print(f"✓ Using GPU {args.gpu_device}: {torch.cuda.get_device_name(args.gpu_device)}")
        else:
            print("CUDA not available, using CPU")
            DEVICE = torch.device("cpu")
    else:
        # Use default: first available GPU or CPU
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda:0")
            print(f"✓ Using default GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            DEVICE = torch.device("cpu")
            print("CUDA not available, using CPU")
    
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Total available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    set_seed(args.seed)
    texts, labels = load_data(args.csv_path, args.text_column, args.label_column)
    
    # Run cross-validation
    averaged_metrics, exp_dir = run_cross_validation(args, texts, labels)
    
    # Train final model on full dataset if requested
    if args.train_final_model:
        train_final_model(
            model_name=args.model_name,
            texts=texts,
            labels=labels,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_path=exp_dir,
        )
        print(f"\n{'='*60}")
        print("Training complete! Model ready for inference.")
        print(f"Model location: {os.path.join(exp_dir, 'final_model')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

