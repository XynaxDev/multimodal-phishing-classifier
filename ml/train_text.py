# ================================================================
# USER CONFIG — EDIT THESE VALUES
# ================================================================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "text", "combined_reduced_splits.csv")
OUTPUT_DIR = "backend/models/text/bert_finetuned/"
MODEL_NAME = "bert-base-uncased"

EPOCHS = 3
LR = 2e-05
WEIGHT_DECAY = 0.01
MAX_LEN = 256
BATCH_SIZE = 16      # RTX 3050 handles this with AMP
VAL_SPLIT = 0.15
SEED = 42
EARLY_STOPPING_PATIENCE = 2
# ================================================================

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# ------------------------------------------------------------
# Custom Dataset
# ------------------------------------------------------------
class PhishTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ------------------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------------------
def main():
    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    df = df.dropna()
    df["text"] = df["text"].astype(str)

    if "split" not in df.columns:
        raise SystemExit("Expected 'split' column. Run ml/build_global_splits.py first.")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = PhishTextDataset(train_df["text"], train_df["label"], tokenizer, MAX_LEN)
    val_ds = PhishTextDataset(val_df["text"], val_df["label"], tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------
    # Device Setup
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ------------------------------------------------------------
    # Load BERT Model
    # ------------------------------------------------------------
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    patience_counter = 0
    print("\n🔥 Training BERT with AMP...\n")

    # ------------------------------------------------------------
    # EPOCH LOOP
    # ------------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        # ------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    preds = outputs.logits.argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss:.4f} | ValAcc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("💾 Saved BETTER BERT model")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("\nTraining complete! Best Accuracy:", best_acc)


if __name__ == "__main__":
    main()
