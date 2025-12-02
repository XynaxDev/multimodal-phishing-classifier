import os

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_IN = os.path.join(BASE_DIR, "data", "text", "combined_reduced.csv")
TEXT_OUT = os.path.join(BASE_DIR, "data", "text", "combined_reduced_splits.csv")

IMAGE_ROOT = "data/images/screenshots"
IMAGE_SPLIT_CSV = "data/images/image_splits.csv"

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def build_text_splits():
    if not os.path.exists(TEXT_IN):
        raise SystemExit(f"Missing text CSV: {TEXT_IN}")

    df = pd.read_csv(TEXT_IN).dropna(subset=["text", "label"])

    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        stratify=df["label"],
        random_state=SEED,
    )

    val_rel = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_rel,
        stratify=trainval_df["label"],
        random_state=SEED,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out_df = pd.concat([train_df, val_df, test_df], axis=0)
    out_df = out_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(TEXT_OUT), exist_ok=True)
    out_df.to_csv(TEXT_OUT, index=False, encoding="utf-8")
    print("Saved text splits:", TEXT_OUT)
    print(out_df["split"].value_counts())


def build_image_splits():
    paths, labels = [], []
    class_map = {"benign": 0, "phishing": 1}

    for cls, lab in class_map.items():
        folder = os.path.join(IMAGE_ROOT, cls)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(lab)

    if not paths:
        print(f"No images found under {IMAGE_ROOT}; skipping image splits.")
        return

    trainval_idx, test_idx = train_test_split(
        range(len(paths)),
        test_size=TEST_FRAC,
        stratify=labels,
        random_state=SEED,
    )

    trainval_labels = [labels[i] for i in trainval_idx]
    val_rel = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_idx_rel, val_idx_rel = train_test_split(
        range(len(trainval_idx)),
        test_size=val_rel,
        stratify=trainval_labels,
        random_state=SEED,
    )

    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]

    import pandas as pd  # local import to avoid confusion if sklearn not present

    rows = []
    for i in train_idx:
        rows.append({"path": paths[i], "label": labels[i], "split": "train"})
    for i in val_idx:
        rows.append({"path": paths[i], "label": labels[i], "split": "val"})
    for i in test_idx:
        rows.append({"path": paths[i], "label": labels[i], "split": "test"})

    df_img = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(IMAGE_SPLIT_CSV), exist_ok=True)
    df_img.to_csv(IMAGE_SPLIT_CSV, index=False, encoding="utf-8")
    print("Saved image splits:", IMAGE_SPLIT_CSV)
    print(df_img["split"].value_counts())


def main():
    build_text_splits()
    build_image_splits()


if __name__ == "__main__":
    main()
