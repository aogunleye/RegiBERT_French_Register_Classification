"""
Curvature landscape figure for RegiBERT, reproducing the method used in
Di Sipio, Diaz-Rodriguez & Serrano (2025), "The Curved Spacetime of Transformer
Architectures" (arXiv:2511.03060) and their reference implementation.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def get_example_tweet(register: str, top_k: int = 1) -> str:
    """text of the top_k most confident tweet for the given register"""
    df = pd.read_csv(config.DATA_PATH, sep="\t")
    if "Poubelle" in df.columns:
        df = df.drop(columns=["Poubelle"])
    return df.sort_values(by=register, ascending=False).iloc[top_k - 1]["Texte"]


def get_layer_matrix(text: str, tokenizer, model, device):
    """ returns:
    tokens: list of str (special tokens excluded), length n_tokens
    layer_matrix: np.ndarray of shape (n_layers=13, n_tokens, hidden_dim=768)
    """
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1).cpu().numpy()  # (13, seq_len, 768)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
               tokenizer.bos_token, tokenizer.eos_token}
    keep_idx = [i for i, t in enumerate(all_tokens) if t not in special]

    tokens = [all_tokens[i].replace("▁", "") for i in keep_idx]
    layer_matrix = hidden_states[:, keep_idx, :]  # (13, n_tokens, 768)
    return tokens, layer_matrix

def turning_angle_matrix(layer_matrix: np.ndarray) -> np.ndarray:
    first = layer_matrix[1:-1] - layer_matrix[:-2]
    second = layer_matrix[2:] - layer_matrix[1:-1]
    first = first / np.maximum(np.linalg.norm(first, axis=2, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=2, keepdims=True), 1e-12)
    cosines = np.clip(np.sum(first * second, axis=2), -1.0, 1.0)
    return np.degrees(np.arccos(cosines))  # shape (n_layers - 2, n_tokens)


def hull_mask(points: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    polygon = MplPath(points[hull.vertices])
    grid_points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    return polygon.contains_points(grid_points).reshape(x_grid.shape)


def interpolate_layer(xy: np.ndarray, values: np.ndarray, grid_res: int):
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res), )
    z_grid = griddata(xy, values, (x_grid, y_grid), method="linear")
    mask = hull_mask(xy, x_grid, y_grid)
    return x_grid, y_grid, np.where(mask, z_grid, np.nan)


def draw_layer(ax, xy: np.ndarray, values: np.ndarray, layer: int, tokens: list,
               vmin: float, vmax: float, grid_res: int):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=90.0, vmax=vmax)
    try:
        x_grid, y_grid, z_grid = interpolate_layer(xy, values, grid_res)
        image = ax.imshow(
            z_grid,
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
        )
        with np.errstate(invalid="ignore"):
            ax.contour(x_grid, y_grid, z_grid, levels=10, colors="k", linewidths=0.35, alpha=0.35)
    except Exception:
        image = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="coolwarm", norm=norm, s=22)

    ax.scatter(xy[:, 0], xy[:, 1], s=8, c="black", alpha=0.45)
    for idx, token in enumerate(tokens):
        ax.annotate(token, xy[idx], fontsize=7, color="black")
    ax.set_title(f"Layer {layer+1}")
    ax.set_xticks([])
    ax.set_yticks([])
    return image

def build_landscape(text: str, tokenizer, model, device,
                     layers_to_show=(3, 6, 9, 12), grid_res=120,
                     vmin=75.0, vmax=105.0, output_path: Path | None = None):
    tokens, layer_matrix = get_layer_matrix(text, tokenizer, model, device)
    n_layers, n_tokens, dim = layer_matrix.shape  # n_layers = 13 for camembert-base

    theta = turning_angle_matrix(layer_matrix)  # (n_layers - 2, n_tokens), covers layers 1..n_layers-2

    pca = PCA(n_components=2, random_state=config.SEED)
    pooled = layer_matrix.reshape(n_layers * n_tokens, dim)
    pca.fit(pooled)
    pca_layers = [pca.transform(layer_matrix[layer]) for layer in range(n_layers)]

    valid_layers = [l for l in layers_to_show if 1 <= l <= n_layers - 2]
    if not valid_layers:
        raise ValueError(f"No valid layers requested. Valid range is 1 to {n_layers - 2}.")

    cols = min(2, len(valid_layers))
    rows = int(np.ceil(len(valid_layers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.2 * rows), squeeze=False)

    last_image = None
    for ax, layer in zip(axes.ravel(), valid_layers):
        values = theta[layer - 1]  # theta index 0 corresponds to layer 1
        last_image = draw_layer(ax, pca_layers[layer], values, layer, tokens, vmin, vmax, grid_res)

    for ax in axes.ravel()[len(valid_layers):]:
        ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82, label="Turning angle (°)")
    fig.suptitle(f'Curvature landscape across CamemBERT layers\n"{text}"', y=0.98, fontsize=10)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
        print(f"Saved {output_path}")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CamemBERT curvature landscape for a TrémoLo tweet.")
    parser.add_argument("--register", default="Soutenu", choices=["Soutenu", "Courant", "Familier"])
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 5, 8, 11])
    parser.add_argument("--output", type=Path, default=Path("images/curvature_landscape.png"))
    parser.add_argument("--grid-res", type=int, default=120)
    parser.add_argument("--vmin", type=float, default=75.0)
    parser.add_argument("--vmax", type=float, default=105.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModel.from_pretrained(config.MODEL_NAME, output_hidden_states=True).to(device)
    model.eval()

    text = get_example_tweet(args.register)
    build_landscape(text, tokenizer, model, device, layers_to_show=args.layers, grid_res=args.grid_res, vmin=args.vmin, vmax=args.vmax, output_path=args.output,)


if __name__ == "__main__":
    main()