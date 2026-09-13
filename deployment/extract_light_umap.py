from pathlib import Path
import joblib
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

input_umap_path = project_root / "checkpoints" / "umap_3d.joblib"
output_dir = current_dir / "checkpoints"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Chargement de {input_umap_path} ...")
umap_data = joblib.load(input_umap_path)

reducer = umap_data["umap_model"]

attrs_to_remove = [
    "_raw_data", 
    "_input_hash", 
    "graph_", 
    "_small_data", 
    "dict_origin_", 
    "_knn_search_index"
]

for attr in attrs_to_remove:
    if hasattr(reducer, attr):
        setattr(reducer, attr, None)

if hasattr(reducer, "_tree") and reducer._tree is not None:
    reducer._tree = None

output_reducer_path = output_dir / "umap_reducer.joblib"
joblib.dump(reducer, output_reducer_path, compress=9)

projections = umap_data["projections"].astype(np.float32)
np.save(output_dir / "static_projections.npy", projections)

if "targets" in umap_data:
    targets = np.array(umap_data["targets"], dtype=np.float32)
    np.save(output_dir / "static_targets.npy", targets)

size_mb = output_reducer_path.stat().st_size / (1024 * 1024)
print(f"Fichier reducer réduit à {size_mb:.2f} Mo !")