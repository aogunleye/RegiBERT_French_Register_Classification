from pathlib import Path
from huggingface_hub import HfApi, create_repo

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

# Upload des 4 fichiers binaires principaux
for fname in files_to_upload:
    path = checkpoints_dir / fname
    if path.exists():
        print(f"Uploading {fname}...")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=fname, repo_id=REPO_ID, repo_type="model")

# Upload du dossier tokenizer
tokenizer_dir = checkpoints_dir / "tokenizer"
if tokenizer_dir.exists():
    print("Uploading tokenizer folder...")
    api.upload_folder(folder_path=str(tokenizer_dir), path_in_repo="tokenizer", repo_id=REPO_ID, repo_type="model")

print("Upload terminé !")