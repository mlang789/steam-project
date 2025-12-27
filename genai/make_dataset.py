# genai/make_dataset.py
"""
Build an instruction fine-tuning dataset from real Steam reviews.

Input:
- data/processed/reviews_with_title.csv

Output:
- reports/sft_train.jsonl
- reports/sft_val.jsonl

Each JSONL row:
{
  "prompt": "...",
  "completion": "..."
}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/reviews_with_title.csv")
OUT_DIR = Path("reports")
TRAIN_PATH = OUT_DIR / "sft_train.jsonl"
VAL_PATH = OUT_DIR / "sft_val.jsonl"

# Keep it aligned with your project choice
POS_RATING = 9
NEG_RATING = 3
ALLOWED_RATINGS = {POS_RATING, NEG_RATING}

MIN_REVIEW_CHARS = 80  # helps training quality
MAX_REVIEW_CHARS = 1200  # avoid very long outliers


def make_sft_prompt(title: str, rating: int) -> str:
    """
    Option A: Fine-tune for Steam-like style + target sentiment (coarse rating),
    without strict formatting constraints that real reviews do not reliably follow.
    """
    sentiment = "positive" if rating >= 7 else "negative"

    return "\n".join([
        "You are a regular Steam user.",
        "",
        f'Write a {sentiment} Steam user review for the game "{title}".',
        f"Target rating: {rating}/10.",
        "",
        "Guidelines:",
        "- Casual, honest tone",
        "- No spoilers",
        "- Output only the review text",
        "",
        "Review:",
    ])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"title", "rating", "review_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["rating"] = df["rating"].astype(int)

    # Filter ratings & basic length constraints
    df = df[df["rating"].isin(ALLOWED_RATINGS)].copy()
    df = df[df["review_text"].str.len() >= MIN_REVIEW_CHARS].copy()
    df = df[df["review_text"].str.len() <= MAX_REVIEW_CHARS].copy()
    df = df[df["title"].str.strip().str.len() > 0].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check your input file.")

    # Create prompt/completion
    rows = []
    for _, r in df.iterrows():
        prompt = make_sft_prompt(r["title"], int(r["rating"]))
        completion = r["review_text"].strip()

        # Optional: enforce "Output only the review text" behavior
        # by ensuring completion starts directly with text.
        rows.append({"prompt": prompt, "completion": completion})

    # Split
    train_rows, val_rows = train_test_split(rows, test_size=0.1, random_state=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)

    print(f"Saved:\n- {TRAIN_PATH} ({len(train_rows)} rows)\n- {VAL_PATH} ({len(val_rows)} rows)")
    print("Example prompt:\n")
    print(train_rows[0]["prompt"][:500])
    print("\nExample completion:\n")
    print(train_rows[0]["completion"][:300])


if __name__ == "__main__":
    main()
