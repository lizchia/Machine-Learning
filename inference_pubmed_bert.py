"""
Inference script for making predictions with a trained PubMedBERT model.

Usage:
    python inference_pubmed_bert.py \
        --model_path ./pubmed_outputs/experiment_20240115_143022/final_model \
        --input_text "Patient has severe fever and high blood pressure."
    
    # Or for batch predictions from CSV:
    python inference_pubmed_bert.py \
        --model_path ./pubmed_outputs/experiment_20240115_143022/final_model \
        --csv_path test_data.csv \
        --text_column text \
        --output_path predictions.csv
"""
from __future__ import annotations

import os
import argparse
from typing import List, Tuple

# IMPORTANT: CUDA_VISIBLE_DEVICES must be set BEFORE importing torch
# To use a specific GPU, set it as an environment variable:
#   export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
#   python inference_pubmed_bert.py ...
# Or run: CUDA_VISIBLE_DEVICES=1 python inference_pubmed_bert.py ...
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Default to GPU 0

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Device will be set in main() based on --gpu_device argument
DEVICE = None


def load_model(model_path: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    print(f"✓ Model loaded successfully on {DEVICE}")
    return model, tokenizer


def predict_single(
    model, tokenizer, text: str, max_len: int = 512, return_probs: bool = False
) -> Tuple[int, float]:
    """
    Predict class for a single text.
    
    Returns:
        (predicted_class, confidence_score)
        If return_probs=True, returns (predicted_class, probability_dict)
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    
    if return_probs:
        prob_dict = {i: float(probs[0][i].item()) for i in range(probs.shape[1])}
        return predicted_class, prob_dict
    
    return predicted_class, confidence


def predict_batch(
    model, tokenizer, texts: List[str], max_len: int = 512, batch_size: int = 32
) -> Tuple[List[int], List[float]]:
    """
    Predict classes for a batch of texts.
    
    Returns:
        (predicted_classes, confidence_scores)
    """
    predictions = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        encodings = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
        batch_confs = torch.max(probs, dim=1)[0].cpu().numpy()
        
        predictions.extend(batch_preds.tolist())
        confidences.extend(batch_confs.tolist())
    
    return predictions, confidences


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained PubMedBERT model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model directory.")
    parser.add_argument("--input_text", type=str, default=None, help="Single text to predict.")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV file with texts to predict.")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing texts in CSV.")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="Output CSV path for batch predictions.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch predictions.")
    parser.add_argument("--return_probs", action="store_true", help="Return probability scores for all classes.")
    parser.add_argument("--gpu_device", type=int, default=None, help="Explicit GPU device index to use (e.g., 0, 1, 2). If not specified, uses first available GPU or CPU.")
    
    args = parser.parse_args()
    
    # Set device explicitly based on user's GPU selection
    global DEVICE
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
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Single prediction
    if args.input_text:
        print(f"\nInput text: {args.input_text}")
        if args.return_probs:
            pred_class, probs = predict_single(model, tokenizer, args.input_text, args.max_len, return_probs=True)
            print(f"Predicted class: {pred_class}")
            print("Class probabilities:")
            for cls, prob in probs.items():
                print(f"  Class {cls}: {prob:.4f}")
        else:
            pred_class, confidence = predict_single(model, tokenizer, args.input_text, args.max_len)
            print(f"Predicted class: {pred_class}")
            print(f"Confidence: {confidence:.4f}")
    
    # Batch prediction from CSV
    elif args.csv_path:
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
        
        df = pd.read_csv(args.csv_path)
        if args.text_column not in df.columns:
            raise ValueError(f"Column '{args.text_column}' not found in CSV.")
        
        texts = df[args.text_column].astype(str).tolist()
        print(f"\nProcessing {len(texts)} texts...")
        
        predictions, confidences = predict_batch(model, tokenizer, texts, args.max_len, args.batch_size)
        
        # Create output dataframe
        output_df = df.copy()
        output_df["predicted_class"] = predictions
        output_df["confidence"] = confidences
        
        # Save predictions
        output_df.to_csv(args.output_path, index=False)
        print(f"✓ Predictions saved to: {args.output_path}")
        print(f"  - Total predictions: {len(predictions)}")
        print(f"  - Class distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    else:
        parser.error("Either --input_text or --csv_path must be provided.")


if __name__ == "__main__":
    main()

