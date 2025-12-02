import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------
# INPUT DATA FILES
# -------------------------
TRANCO_CSV      = "data/url_raw/benign.csv"
URLHAUS_CSV     = "data/url_raw/urlhaus_malware.csv"
PHISHTANK_CSV   = "data/url_raw/phishtank.csv"

OUT_FULL_CSV    = "data/url_dataset/url_multiclass_full.csv"
OUT_SPLIT_CSV   = "data/url_dataset/url_multiclass_splits.csv"

TARGET_PER_CLASS = 50000   # OR set to None to use all
SEED = 42


# -------------------------
# URL NORMALIZER
# -------------------------
def normalize_url(u: str):
    if not isinstance(u, str):
        return ""
    u = u.strip().strip('"').strip("'")

    # add https:// if missing
    if not re.match(r"^\w+://", u):
        u = "https://" + u

    # lowercase everything
    return u.lower()


# -------------------------
# LOAD BENIGN (class 0)
# -------------------------
def load_benign(path):
    df = pd.read_csv(path, header=None)

    # Find the domain column automatically
    domain_col = None
    for col in df.columns:
        if df[col].astype(str).str.contains(r"\.", regex=True).any():
            domain_col = col
            break
    if domain_col is None:
        raise SystemExit("❌ Could not detect domain column in Tranco benign dataset")

    urls = df[domain_col].astype(str).tolist()
    urls = [normalize_url(u) for u in urls]
    urls = list(set(urls))  # dedupe

    return pd.DataFrame({"url": urls, "label": 0})


# -------------------------
# LOAD URLHAUS MALWARE (class 2)
# -------------------------
def load_urlhaus(path):
    df = pd.read_csv(path)
    df["url"] = df["url"].astype(str)

    df = df[["url"]]  # only URL needed
    df["label"] = 2   # malware class

    df["url"] = df["url"].apply(normalize_url)
    df = df.dropna().drop_duplicates()

    return df


# -------------------------
# LOAD PHISHTANK PHISHING (class 1)
# -------------------------
def load_phishtank(path):
    df = pd.read_csv(path)

    if "url" not in df.columns:
        raise SystemExit("❌ PhishTank CSV missing 'url' column")

    df["url"] = df["url"].astype(str)
    df = df[["url"]]
    df["label"] = 1  # phishing

    df["url"] = df["url"].apply(normalize_url)
    df = df.dropna().drop_duplicates()

    return df


# -------------------------
# BUILD FINAL DATASET
# -------------------------
def main():
    os.makedirs("data/url_dataset", exist_ok=True)

    print("\n🔵 Loading benign...")
    benign_df = load_benign(TRANCO_CSV)
    print("Benign count:", len(benign_df))

    print("\n🔴 Loading malware (URLHaus)...")
    malware_df = load_urlhaus(URLHAUS_CSV)
    print("Malware count:", len(malware_df))

    print("\n🟡 Loading phishing (PhishTank)...")
    phish_df = load_phishtank(PHISHTANK_CSV)
    print("Phishing count:", len(phish_df))

    # -------------------------
    # BALANCE CLASSES
    # -------------------------
    def balance(df, target):
        if target is None:
            return df
        return df.sample(min(len(df), target), random_state=SEED)

    benign_df  = balance(benign_df, TARGET_PER_CLASS)
    phish_df   = balance(phish_df, TARGET_PER_CLASS)
    malware_df = balance(malware_df, TARGET_PER_CLASS)

    print("\nFinal class sizes:")
    print("Benign:   ", len(benign_df))
    print("Phishing: ", len(phish_df))
    print("Malware:  ", len(malware_df))

    # -------------------------
    # MERGE + SHUFFLE
    # -------------------------
    final_df = pd.concat([benign_df, phish_df, malware_df], axis=0)
    final_df = final_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # save main CSV
    final_df.to_csv(OUT_FULL_CSV, index=False)
    print("\n[✓] Saved full dataset:", OUT_FULL_CSV)
    print("Total rows:", len(final_df))

    # -------------------------
    # SPLITS: train/val/test
    # -------------------------
    train_val, test = train_test_split(
        final_df,
        test_size=0.15,
        stratify=final_df["label"],
        random_state=SEED
    )

    val_ratio = 0.15 / 0.85
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["label"],
        random_state=SEED
    )

    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    split_df = pd.concat([train, val, test], axis=0)
    split_df = split_df.sample(frac=1.0, random_state=SEED)

    split_df.to_csv(OUT_SPLIT_CSV, index=False)
    print("\n[✓] Saved split dataset:", OUT_SPLIT_CSV)

    print("\nSplit counts:")
    print(split_df["split"].value_counts())
    print(split_df.head())


if __name__ == "__main__":
    main()
