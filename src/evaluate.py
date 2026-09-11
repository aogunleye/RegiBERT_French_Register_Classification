import sys
from pathlib import Path

from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

import config
from dataset import get_dataloaders
from model import RegiBERT


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
                logprobs, _ = model(input_ids, attention_mask)
                loss = criterion(logprobs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            probs = torch.exp(logprobs)

            # save predictions and targets for metrics calculation
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # stacking batches (shape: [samples, classes])
    all_preds = np.vstack(all_preds) 
    all_targets = np.vstack(all_targets)

    avg_loss = total_loss/len(dataloader)
    
    
    return avg_loss, all_preds, all_targets


def compute_metrics(preds, targets):
    classes = ["Soutenu", "Courant", "Familier"]
    
    mae_per_class = mean_absolute_error(targets, preds, multioutput="raw_values")
    total_mae = mean_absolute_error(targets, preds)

    pred_labels = np.argmax(preds, axis=1)
    target_labels = np.argmax(targets, axis=1)
    accuracy = accuracy_score(target_labels, pred_labels)

    return mae_per_class, total_mae, accuracy, classes


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    _, val_loader, _ = get_dataloaders()

    model_path = config.MODEL_SAVE_PATH
    model = RegiBERT()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Unable to find the file {model_path}. Please run train.py first.")
        return

    model.to(device)

    criterion = nn.KLDivLoss(reduction="batchmean")
    val_loss, preds, targets = evaluate_model(model, val_loader, criterion, device)
    
    mae_per_class, total_mae, accuracy, classes = compute_metrics(preds, targets)

    print("\n--- Evaluation metrics --- ")
    print(f"KL Divergence Loss : {val_loss:.4f}")
    print(f"Global MAE         : {total_mae:.4f}")
    print(f"Accuracy           : {accuracy * 100:.2f}%\n")

    print("--- MAE per register ---")
    for class_name, mae_val in zip(classes, mae_per_class):
        print(f"    - {class_name:<10} : {mae_val:.4f}")

if __name__ == "__main__":
    main()