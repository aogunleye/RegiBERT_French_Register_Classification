import os
import torch

# based on my hardware and dataset size, but can be adjusted
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"  # activates fp16 mixed precision if using GPU
MODEL_NAME = "camembert-base"
NUM_CLASSES = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_WORKERS = 4 
EPOCHS = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tremolo.tsv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
UMAP_SAVE_PATH = os.path.join(BASE_DIR, "checkpoints", "umap_3d.joblib")