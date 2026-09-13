"""
Uploads the deployment checkpoint files to a Hugging Face Hub repo, so they
never need to go through git (avoids GitHub's 100MB file limit).

Run once locally, after export_to_onnx.py / quantize_onnx.py /
extract_neighbors_data.py have produced the 4 files.

Requires: pip install huggingface_hub --break-system-packages
You also need to be logged in: huggingface-cli login (or set HF_TOKEN env var)
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

# CHANGE THIS to your own HF username/repo name
REPO_ID = "aogunleye/regibert-deployment"

checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
files_to_upload = [
    "regibert_int8.onnx",
    "static_embeddings.npy",
    "static_projections.npy",
    "static_targets.npy",
]

api = HfApi()
create_repo(REPO_ID, repo_type="model", exist_ok=True)

for fname in files_to_upload:
    path = checkpoints_dir / fname
    print(f"Uploading {fname} ({path.stat().st_size / (1024*1024):.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="model",
    )

print(f"\nDone. Files available at: https://huggingface.co/{REPO_ID}")