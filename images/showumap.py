from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

checkpoint_path = Path("checkpoints/umap_3d.joblib")
data = joblib.load(checkpoint_path)

coords = data["projections"] 
targets = data["targets"]    

colors = np.zeros((len(targets), 3))
colors[:, 0] = targets[:, 0]  
colors[:, 1] = targets[:, 2] 
colors[:, 2] = targets[:, 1] 

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(projection="3d")

ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=2, alpha=0.5)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.set_title("Calibrated 3D UMAP Manifold on validation set")

plt.tight_layout()

output_path = Path("images/umap_projection_static.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)

print(f"Saved : {output_path}")
plt.show()