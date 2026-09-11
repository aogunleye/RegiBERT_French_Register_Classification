import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
import config
from dataset import get_dataloaders
from model import RegiBERT
from tqdm import tqdm 


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()

        # forward pass
        with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
            logprobs, _ = model(input_ids, attention_mask)
            loss = criterion(logprobs, targets)

        # backpropagation with clipping 
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        # update learning rate scheduler only if the scale hasn't changed (to avoid skipping steps when gradients are too large)
        if scheduler is not None and scale_before <= scale_after:
            scheduler.step()

        total_loss += loss.item()
        
        # Affichage dynamique dans tqdm
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
                logprobs, _ = model(input_ids, attention_mask)
                loss = criterion(logprobs, targets)

            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    train_loader, val_loader, tokenizer = get_dataloaders()
    model = RegiBERT().to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    steps = len(train_loader)*config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps*0.1),num_training_steps=steps)

    # training loop with early stopping based on validation loss
    best_val_loss = float("inf")
    save_path = config.MODEL_SAVE_PATH
    os.makedirs("checkpoints", exist_ok=True)

    scaler = GradScaler(enabled=config.USE_AMP)


    print("\nStarting training...\n")
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path}")

    print("\nTraining completed. Best validation loss: {:.4f}".format(best_val_loss))

if __name__ == "__main__":
    main()