# src/build_reviews_with_title.py
"""
Build an enriched review dataset with:
- game title (from data/processed/games.csv)
- a numeric rating derived from recommended (Steam is binary)

Input:
- data/processed/reviews.csv
- data/processed/games.csv

Output:
- data/processed/reviews_with_title.csv
"""

from pathlib import Path

import pandas as pd

REVIEWS_PATH = Path("data/processed/reviews.csv")
GAMES_PATH = Path("data/processed/games.csv")
OUT_PATH = Path("data/processed/reviews_with_title.csv")

# Coarse rating mapping (defendable and simple)
RATING_POS = 9
RATING_NEG = 3


def main() -> None:
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {REVIEWS_PATH}")
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {GAMES_PATH}")

    reviews = pd.read_csv(REVIEWS_PATH)
    games = pd.read_csv(GAMES_PATH)

    # Basic column checks
    required_reviews_cols = {"app_id", "recommended", "review_text"}
    required_games_cols = {"app_id", "title"}

    missing_reviews = required_reviews_cols - set(reviews.columns)
    missing_games = required_games_cols - set(games.columns)

    if missing_reviews:
        raise ValueError(f"reviews.csv missing columns: {sorted(missing_reviews)}")
    if missing_games:
        raise ValueError(f"games.csv missing columns: {sorted(missing_games)}")

    # Merge titles
    df = reviews.merge(games, on="app_id", how="left")

    # Sanity check: any missing titles?
    missing_titles = df["title"].isna().sum()
    if missing_titles > 0:
        print(f"Warning: {missing_titles} reviews have missing titles after merge.")

    # Ensure types
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # Create numeric rating (Steam has only voted_up)
    df["recommended"] = df["recommended"].astype(int)
    df["rating"] = df["recommended"].map({1: RATING_POS, 0: RATING_NEG}).astype(int)

    # Reorder columns (nice for downstream gen/eval)
    front_cols = ["app_id", "title", "rating", "recommended", "review_text"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Saved {OUT_PATH} with shape: {df.shape}")
    print("Preview:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
