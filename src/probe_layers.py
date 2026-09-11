import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import config
from dataset import get_dataloaders
from model import RegiBERT


def extract_all_layers_embeddings(model, dataloader, device, max_samples=10000):
    """extracts embeddings from all 12 layers of CamemBERT for a given dataset and returns them along with the corresponding targets"""
    model.eval()
    
    layer_embeddings = {layer_idx: [] for layer_idx in range(13)} # 13 layers (12 hidden+ 1 input embedding layer)
    all_targets = []
    total_samples = 0

    print(f"Extracting embeddings from all layers for up to {max_samples} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            if total_samples >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"]

            with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
                outputs = model.camembert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of 13 tensors (1 for input embeddings + 12 for hidden layers)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states[0].size()).float()

            for layer_idx in range(13):
                layer_state = hidden_states[layer_idx]
                sum_embeddings = torch.sum(layer_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).float().cpu().numpy() # for cpu processing, to avoid memory issues and ensure float32 precision
                
                layer_embeddings[layer_idx].append(mean_pooled)

            all_targets.append(targets.numpy())
            total_samples += input_ids.size(0) # increment by the batch size

    for layer_idx in range(13):
        layer_embeddings[layer_idx] = np.vstack(layer_embeddings[layer_idx])[:max_samples]
    
    all_targets = np.vstack(all_targets)[:max_samples]

    return layer_embeddings, all_targets


def evaluate_layer_separability(layer_embeddings, targets):
    """computes the R² score for each layer's embeddings using Ridge regression to evaluate geometric separability of the classes"""
    scores = {}
    print("\nEvaluating geometric separability of classes for each layer...")

    for layer_idx in tqdm(range(13), desc="Evaluating layers"):
        X = layer_embeddings[layer_idx]
        y = targets

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # ridge with cross-validation to find the best alpha
        ridge = RidgeCV()
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_val)

        r2 = r2_score(y_val, preds, multioutput="variance_weighted")
        scores[layer_idx] = r2
        print(f"  - Layer {layer_idx} | R² Score : {r2:.4f} (best alpha: {ridge.alpha_})")

    best_layer = max(scores, key=scores.get)
    return scores, best_layer 


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    _, val_loader, _ = get_dataloaders()

    model = RegiBERT()
    model_path = config.MODEL_SAVE_PATH

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Unable to find {model_path}.")
        return

    model.to(device)

    layer_embeddings, targets = extract_all_layers_embeddings(model, val_loader, device, max_samples=10000)
    scores, best_layer = evaluate_layer_separability(layer_embeddings, targets)

    best_layer_file = Path(config.MODEL_SAVE_PATH).parent / "best_layer.json"
    with open(best_layer_file, "w") as f:
        json.dump({"best_layer": int(best_layer), "r2_score": float(scores[best_layer])}, f)

    print(f"Best layer is layer {best_layer} with R² = {scores[best_layer]:.4f}")
    print(f"Keeping {best_layer} as the best layer for fit_umap.py")


if __name__ == "__main__":
    main()