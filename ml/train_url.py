# ml/train_url_classifier.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------- CONFIG -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "url_dataset", "url_multiclass_splits.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "models", "url", "distilbert_url")

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 1                   
BATCH_SIZE = 16               
MAX_LEN = 64                  
LR = 1e-5 
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2   # <--- you had commented it, it must exist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# ------------------- CUSTOM DATASET -------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, maxlen):
        self.urls = list(urls)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.urls[idx],
            truncation=True,
            padding="max_length",
            max_length=self.maxlen,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ----------------------- MAIN -------------------------
def main():
    print("📥 Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = URLDataset(train_df["url"], train_df["label"], tokenizer, MAX_LEN)
    val_ds = URLDataset(val_df["url"], val_df["label"], tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ================================
    # CREATE MODEL FIRST  (IMPORTANT)
    # ================================
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    # ----------------------------------------------
    # NOW apply dropout to config (AFTER model exists)
    # ----------------------------------------------
    model.config.dropout = 0.3
    model.config.attention_dropout = 0.3

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0
    patience = 0

    print("\n🚀 Starting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (len(pbar)+1)})

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=mask).logits

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"📊 Epoch {epoch+1} | Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print("💾 Saved BEST model")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("⛔ Early stopping")
                break

    print("\n🎉 Training done! Best Val Acc:", best_val_acc)

if __name__ == "__main__":
    main()
