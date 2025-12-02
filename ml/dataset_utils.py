import json
import pandas as pd
import os

os.makedirs("data/text", exist_ok=True)

# Load JSON file
with open("data/raw/combined_reduced.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure only required columns
df = df[['text', 'label']].dropna()

# Save CSV
df.to_csv("data/text/combined_reduced.csv", index=False, encoding="utf-8")

print("Saved:", "data/text/combined_reduced.csv")
print("Shape:", df.shape)
