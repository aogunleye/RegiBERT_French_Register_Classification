import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
import config

class TremoloDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        return {"input_ids": encoding["input_ids"].squeeze(0),"attention_mask": encoding["attention_mask"].squeeze(0),"labels": label}


def get_dataloaders():
    df = pd.read_csv(config.DATA_PATH, sep='\t')
    if "Poubelle" in df.columns:
        df = df.drop(columns=["Poubelle"])

    texts = df["Texte"]
    label_cols = ["Soutenu", "Courant", "Familier"]
    labels = df[label_cols]

    # split train/validation 80/20
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = TremoloDataset(train_texts, train_labels, tokenizer)
    val_dataset = TremoloDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,       
    pin_memory=True if torch.cuda.is_available() else False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False)

    return train_loader, val_loader, tokenizer