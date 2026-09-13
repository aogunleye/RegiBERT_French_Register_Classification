import joblib
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

input_umap_path = project_root / "checkpoints" / "umap_3d.joblib"
output_dir = current_dir / "checkpoints"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Chargement de {input_umap_path} ...")
umap_data = joblib.load(input_umap_path)

reducer = umap_data["umap_model"]
joblib.dump(reducer, output_dir / "umap_reducer.joblib", compress=3)

projections = umap_data["projections"].astype(np.float32)
np.save(output_dir / "static_projections.npy", projections)

if "targets" in umap_data:
    targets = np.array(umap_data["targets"], dtype=np.float32)
    np.save(output_dir / "static_targets.npy", targets)

print(f"Extraction terminée avec succès dans {output_dir} !")