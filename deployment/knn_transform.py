from pathlib import Path
import numpy as np

class NeighborProjector:
    def __init__(self, checkpoints_dir: Path, k: int = 15):
        self.k = k

        # 1. Lecture via memory-mapping (0 Mo alloué en mémoire au chargement)
        raw_embeddings = np.load(checkpoints_dir / "static_embeddings.npy", mmap_mode="r")

        # 2. Conversion progressive et directe en float16 pour économiser la RAM
        embeddings = raw_embeddings[:].astype(np.float16)
        
        # 3. Normalisation vectorielle directement en float16
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-4, None)
        self._normalized = embeddings / norms

        del embeddings
        del raw_embeddings

        self.projections = np.load(checkpoints_dir / "static_projections.npy").astype(np.float32)

    def transform(self, query_embedding: np.ndarray) -> list:
        query = query_embedding.reshape(-1).astype(np.float32)
        query_norm = (query / max(np.linalg.norm(query), 1e-9)).astype(np.float16)

        similarities = (self._normalized @ query_norm).astype(np.float32)

        top_k_idx = np.argpartition(-similarities, self.k)[: self.k]
        top_k_sims = similarities[top_k_idx]

        weights = np.clip(top_k_sims, 1e-6, None)
        weights = weights / weights.sum()

        coords = (self.projections[top_k_idx] * weights[:, None]).sum(axis=0)
        return coords.tolist()