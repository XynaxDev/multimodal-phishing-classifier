import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_CSV = os.path.join(BASE_DIR, "data/url_dataset/url_multiclass_splits.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend/models/url/distilbert_url")

MAX_LEN = 64
BATCH_SIZE = 64
FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.urls[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    df = pd.read_csv(SPLIT_CSV)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print("📌 Loaded TEST samples:", len(df_test))

    X = df_test["url"].tolist()
    y = df_test["label"].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    fold_acc = []
    fold_f1 = []

    for fold, (_, idx) in enumerate(skf.split(X, y), 1):
        print(f"\n========== Fold {fold}/{FOLDS} ==========")

        X_fold = [X[i] for i in idx]
        y_fold = [y[i] for i in idx]

        ds = URLDataset(X_fold, y_fold, tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)

        preds, labs = [], []

        with torch.no_grad():
            for batch in tqdm(dl, desc=f"Fold {fold} inference"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(ids, attention_mask=mask).logits
                p = logits.argmax(dim=1).cpu().numpy()

                preds.extend(p)
                labs.extend(labels.cpu().numpy())

        acc = accuracy_score(labs, preds)
        f1 = f1_score(labs, preds, average="weighted")

        fold_acc.append(acc)
        fold_f1.append(f1)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(labs, preds))

    print("\n====== FINAL CV RESULTS ======")
    print("Mean Accuracy:", np.mean(fold_acc))
    print("Mean F1 Score:", np.mean(fold_f1))

if __name__ == "__main__":
    main()
