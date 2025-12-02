import os
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from transformers import BertTokenizerFast, BertForSequenceClassification

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "text", "combined_reduced_splits.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models", "text", "bert_finetuned")
OUT_DIR = os.path.join(BASE_DIR, "runs", "text")

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using:", device)

# =====================================================
# DATASET
# =====================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

# =====================================================
# K-FOLD CV
# =====================================================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

all_acc = []
all_f1 = []

fold_num = 1

for train_idx, test_idx in kfold.split(texts):
    print(f"\n========== Fold {fold_num}/5 ==========")

    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load your existing model (NOT retraining)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold_num} inference"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            logits = model(ids, mask).logits
            preds = logits.argmax(dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(labels_all, preds_all))

    all_acc.append(acc)
    all_f1.append(f1)

    # Confusion Matrix Plot
    cm = confusion_matrix(labels_all, preds_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Fold {fold_num} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"confusion_fold{fold_num}.png"))
    plt.close()

    fold_num += 1

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n===== FINAL 5-FOLD RESULTS =====")
print("Mean Accuracy:", np.mean(all_acc))
print("Mean F1:", np.mean(all_f1))

with open(os.path.join(OUT_DIR, "cv_results.txt"), "w") as f:
    f.write(f"Mean Accuracy: {np.mean(all_acc)}\n")
    f.write(f"Mean F1: {np.mean(all_f1)}\n")
    f.write("Accuracies per fold:\n")
    for a in all_acc:
        f.write(str(a) + "\n")
