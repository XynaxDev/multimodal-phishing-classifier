import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/url_dataset/url_multiclass_splits.csv")

MODEL_DIR = os.path.join(BASE_DIR, "backend/models/url/distilbert_url/")
SAVE_DIR = os.path.join(BASE_DIR, "runs/url_eval")
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LEN = 64
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -------- Dataset --------
class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer):
        self.urls = list(urls)
        self.labels = list(labels)
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
test_df = df[df["split"] == "test"]

print(f"📊 Test samples: {len(test_df)}")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

test_ds = URLDataset(test_df["url"], test_df["label"], tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# ---------------- RUN EVALUATION ----------------
all_labels = []
all_preds = []
all_probs = []

print("\n🔍 Running evaluation...\n")

with torch.no_grad():
    for batch in tqdm(test_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)


all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ---------------- METRICS ----------------
acc = accuracy_score(all_labels, all_preds)

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=[0, 1, 2], zero_division=0
)

report = classification_report(
    all_labels, all_preds,
    target_names=["benign", "phishing", "malware"],
    output_dict=True
)

cm = confusion_matrix(all_labels, all_preds).tolist()

# ---------------- ROC CURVES ----------------
roc_results = {}
num_classes = all_probs.shape[1]

plt.figure(figsize=(7, 6))
for cls in range(num_classes):
    y_true = (all_labels == cls).astype(int)
    y_score = all_probs[:, cls]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    roc_results[f"class_{cls}"] = float(auc_score)

    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves (Multiclass)")
plt.legend()
roc_path = os.path.join(SAVE_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=200)
plt.close()

# ---------------- CONFUSION MATRIX PLOT ----------------
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, str(cm[i][j]), ha="center", va="center", color="red")

cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=200)
plt.close()

# ---------------- CONFIDENCE HISTOGRAM ----------------
plt.figure(figsize=(7, 5))
plt.hist(all_probs.max(axis=1), bins=30, alpha=0.7)
plt.title("Prediction Confidence Histogram")
plt.xlabel("Max class probability")
plt.ylabel("Count")
hist_path = os.path.join(SAVE_DIR, "confidence_histogram.png")
plt.tight_layout()
plt.savefig(hist_path, dpi=200)
plt.close()

# ---------------- FINAL RESULTS JSON ----------------
final_results = {
    "overall_accuracy": float(acc),

    "per_class": {
        "benign": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "phishing": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "malware": {
            "precision": float(precision[2]),
            "recall": float(recall[2]),
            "f1": float(f1[2]),
            "support": int(support[2]),
        },
    },

    "macro_f1": float(report["macro avg"]["f1-score"]),
    "weighted_f1": float(report["weighted avg"]["f1-score"]),

    "confusion_matrix": cm,
    "roc_auc_per_class": roc_results,

    "confidence_stats": {
        "mean_confidence": float(np.mean(all_probs.max(axis=1))),
        "median_confidence": float(np.median(all_probs.max(axis=1))),
        "min_confidence": float(np.min(all_probs.max(axis=1))),
        "max_confidence": float(np.max(all_probs.max(axis=1))),
    },

    "num_test_samples": len(test_df),
    "prediction_distribution": {
        "pred_0_benign": int((all_preds == 0).sum()),
        "pred_1_phishing": int((all_preds == 1).sum()),
        "pred_2_malware": int((all_preds == 2).sum()),
    },

    "paths": {
        "roc_curve_png": roc_path,
        "confusion_matrix_png": cm_path,
        "confidence_hist_png": hist_path,
    }
}

results_path = os.path.join(SAVE_DIR, "final_url_results.json")
with open(results_path, "w") as f:
    json.dump(final_results, f, indent=4)

print("\n🎉 Evaluation complete!")
print(f"📁 Saved full results JSON → {results_path}")
print(f"📁 ROC → {roc_path}")
print(f"📁 Confusion Matrix → {cm_path}")
print(f"📁 Histogram → {hist_path}")
