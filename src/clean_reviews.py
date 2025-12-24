from pathlib import Path
import pandas as pd

# Paths
RAW_PATH = "data/raw/reviews_raw.csv"
PROCESSED_PATH = "data/processed/reviews.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_PATH)

# 1) Keep English reviews only
df["language"] = df["language"].fillna("")
df = df[df["language"] == "english"].copy()

# 2) Remove empty or very short reviews
df["review_text"] = df["review_text"].fillna("").astype(str)
df["text_length"] = df["review_text"].str.len()
df = df[df["text_length"] >= 30].copy()

# 3) Drop duplicates (same text for same game)
df = df.drop_duplicates(subset=["app_id", "review_text"]).copy()

# 4) Select useful columns
columns_to_keep = [
    "app_id",
    "recommended",
    "review_text",
    "timestamp_created",
    "playtime_at_review",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
]
df = df[columns_to_keep].copy()

# Save cleaned dataset
df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")

print(f"Saved {PROCESSED_PATH} with shape: {df.shape}")
print("\nRecommended label distribution:")
print(df["recommended"].value_counts())
