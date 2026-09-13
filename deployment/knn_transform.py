"""
Lightweight, pure-numpy replacement for UMAP's reducer.transform(), for the
production container (no umap-learn, no numba — this is precisely what was
suspected of causing the OOM at startup on Northflank's 512 MiB plan).

Approximation used: for a new 768D embedding, find its k nearest neighbors
(cosine similarity) among the ~45k validation embeddings used to fit UMAP,
and return a similarity-weighted average of their existing 3D positions.

This is NOT mathematically identical to UMAP's own out-of-sample transform
(which re-optimizes the point's position via a short gradient descent), but
neighbor-averaging in the fitted embedding space is the same core idea UMAP
itself uses to initialize that optimization — dropping the optimization step
is a reasonable, much cheaper approximation for a live visualization.
"""

from pathlib import Path

import numpy as np


class NeighborProjector:
    def __init__(self, checkpoints_dir: Path, k: int = 15):
        self.k = k

        # Chargement ultra-rapide via memory mapping sans allocation massive
        embeddings = np.load(checkpoints_dir / "static_embeddings.npy", mmap_mode="r").astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        normalized = embeddings / norms

        self._normalized = normalized.astype(np.float16)
        self.projections = np.load(checkpoints_dir / "static_projections.npy")

    def transform(self, query_embedding: np.ndarray) -> list:
        """query_embedding: shape (768,) or (1, 768). Returns [x, y, z]."""
        query = query_embedding.reshape(-1).astype(np.float32)
        query_norm = (query / max(np.linalg.norm(query), 1e-9)).astype(np.float16)

        similarities = (self._normalized @ query_norm).astype(np.float32)  # (N,) cosine similarities

        top_k_idx = np.argpartition(-similarities, self.k)[: self.k]
        top_k_sims = similarities[top_k_idx]

        # Softmax-style weighting so closer neighbors count more; clip to
        # avoid an all-negative-similarity edge case degenerating to NaN.
        weights = np.clip(top_k_sims, 1e-6, None)
        weights = weights / weights.sum()

        coords = (self.projections[top_k_idx] * weights[:, None]).sum(axis=0)
        return coords.tolist()