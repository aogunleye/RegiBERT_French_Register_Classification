import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import json
import torch
import joblib
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from sklearn.manifold import trustworthiness
from umap import UMAP
import config
from dataset import get_dataloaders
from model import RegiBERT

def extract_best_layer_embeddings(model, dataloader, device, layer_idx):
    model.eval()
    all_embeddings = []
    all_targets = []

    print(f"Extracting embeddings from layer {layer_idx}...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting layer {layer_idx}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"]

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.USE_AMP):
                outputs = model.camembert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            token_embeddings = hidden_states[layer_idx]

            input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).float().cpu().numpy()

            all_embeddings.append(mean_pooled)
            all_targets.append(targets.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.vstack(all_targets)

    return all_embeddings, all_targets

def find_best_n_neighbors(embeddings, candidate_neighbors, seed):
    """tests different n_neighbors values for UMAP and selects the one with the highest trustworthiness score"""
    best_score = float('-inf')
    best_n = candidate_neighbors[0]

    # compute the number of samples for 99% confidence interval with error of 0.01
    sample_size = int((norm.ppf(1 - (1 - 0.99) / 2) * 0.5 / 0.01) ** 2)

    # subset for faster evaluation
    np.random.seed(seed)
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        sub_embeddings = embeddings[indices]
    else:
        sub_embeddings = embeddings

    print(f"\nSearching for the optimal n_neighbors (evaluated on {len(sub_embeddings)} sampled points)...")
    
    for n in tqdm(candidate_neighbors, desc="Testing n_neighbors"):
        reducer = UMAP(n_components=3, metric="cosine", n_neighbors=n, min_dist=0.1, random_state=seed)
        proj = reducer.fit_transform(sub_embeddings)
        
        score = trustworthiness(sub_embeddings, proj, n_neighbors=n, metric="cosine")
        tqdm.write(f"  - n_neighbors = {n:2d}, trustworthiness = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_n = n
            
    print(f"Best n_neighbors = {best_n} (score: {best_score:.4f})\n")
    return best_n, best_score


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    best_layer_file = Path(config.MODEL_SAVE_PATH).parent / "best_layer.json"
    
    if not best_layer_file.exists():
        print(f"Error : {best_layer_file} not found. Please run probe_layers.py first")
        return

    with open(best_layer_file, "r") as f:
        data = json.load(f)
        best_layer = data["best_layer"]
        r2_score = data["r2_score"]

    _, val_loader, _ = get_dataloaders()
    model = RegiBERT()
    model_path = config.MODEL_SAVE_PATH

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from : {model_path}")
    except FileNotFoundError:
        print(f"Error : Unable to find {model_path}")
        return

    model.to(device)

    embeddings, targets = extract_best_layer_embeddings(model, val_loader, device, layer_idx=best_layer)

    candidates = [5, 10, 15, 20, 30]
    neighbors, trust_score = find_best_n_neighbors(embeddings, candidate_neighbors=candidates, seed=config.SEED)

    print(f"Calibrating final UMAP 3D on all {len(embeddings)} points (n_neighbors={neighbors})...")
    final_reducer = UMAP(n_components=3, metric="cosine", n_neighbors=neighbors, min_dist=0.1, random_state=config.SEED)
    final_proj = final_reducer.fit_transform(embeddings)

    os.makedirs(os.path.dirname(config.UMAP_SAVE_PATH), exist_ok=True)
    save_data = {"umap_model": final_reducer, "projections": final_proj, "targets": targets,"best_layer": best_layer, "best_n_neighbors": neighbors, "trustworthiness": trust_score}

    joblib.dump(save_data, config.UMAP_SAVE_PATH)
    print(f"\nProjections and UMAP model saved to : {config.UMAP_SAVE_PATH}")

if __name__ == "__main__":
    main()