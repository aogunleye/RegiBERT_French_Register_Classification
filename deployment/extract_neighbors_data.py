"""
OFFLINE script — run this locally (with PyTorch installed), NOT in the
production container. It produces the three lightweight .npy files the
production server needs to approximate UMAP's out-of-sample transform
without depending on umap-learn/numba at runtime.

Replaces extract_light_umap.py + umap_reducer.joblib entirely.

Run from the project root:
    python deployment/extract_neighbors_data.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.dataset import get_dataloaders
from src.model import RegiBERT


def main():
    device = config.DEVICE
    if isinstance(device, str):
        device = torch.device(device)
    output_dir = Path(__file__).resolve().parent / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Re-extract the 768D pooled embeddings for the validation set ---
    # (these were never persisted by fit_umap.py, only the 3D projections were)
    print("Loading model and validation set...")
    _, val_loader, _ = get_dataloaders()
    model = RegiBERT()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Extracted embeddings: {embeddings.shape}")

    # --- Reuse the existing 3D projections + targets from umap_3d.joblib ---
    umap_path = Path(config.UMAP_SAVE_PATH)
    print(f"Loading projections from {umap_path} ...")
    umap_data = joblib.load(umap_path)
    projections = np.asarray(umap_data["projections"], dtype=np.float32)
    targets = np.asarray(umap_data["targets"], dtype=np.float32)

    assert len(embeddings) == len(projections), (
        f"Mismatch: {len(embeddings)} embeddings vs {len(projections)} projections. "
        "Did the validation split change (different random seed / data) since fit_umap.py ran?"
    )

    np.save(output_dir / "static_embeddings.npy", embeddings)
    np.save(output_dir / "static_projections.npy", projections)
    np.save(output_dir / "static_targets.npy", targets)

    total_mb = sum(
        (output_dir / f).stat().st_size for f in
        ["static_embeddings.npy", "static_projections.npy", "static_targets.npy"]
    ) / (1024 * 1024)
    print(f"Saved 3 files to {output_dir}, total {total_mb:.1f} MB")


if __name__ == "__main__":
    main()