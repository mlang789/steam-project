import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import FILES, GENAI_INPUTS_DIR 

DATA_PATH = FILES["train"]["final"] 
OUT_PATH = GENAI_INPUTS_DIR / "prompt_batch.csv" 

# How many prompts per game per rating (positive/negative)
N_PER_GAME_PER_RATING = 1000

# Ratings used in your dataset
POS_RATING = 9
NEG_RATING = 3


def make_naive_prompt(title: str, rating: int) -> str:
    return (
        f'Write a Steam user review for the game "{title}" with rating {rating}/10.'
    )


def make_engineered_prompt(title: str, rating: int) -> str:
    # Use different constraints depending on target sentiment
    if rating >= 7:
        constraint_line = "- Mention exactly 2 positive points and 1 negative point"
    else:
        constraint_line = "- Mention exactly 2 negative points and 1 positive point"

    return "\n".join([
        "You are a regular Steam user.",
        "",
        f'Write a review for the game "{title}".',
        f"Target rating: {rating}/10.",
        "",
        "Constraints:",
        "- Length: 120 to 140 words (aim for ~125-135)",

        constraint_line,
        "- Casual, honest tone",
        "- No spoilers",
        "- Output only the review text",
    ])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"app_id", "title", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    # Keep only the ratings we expect (3 and 9)
    df = df[df["rating"].isin([NEG_RATING, POS_RATING])].copy()

    rows = []

    # Sample per (game, rating)
    for (app_id, title, rating), group in df.groupby(["app_id", "title", "rating"], dropna=False):
        sample_n = min(N_PER_GAME_PER_RATING, len(group))
        if sample_n == 0:
            continue

        # We don't actually need the review text here, only (title, rating)
        sampled = group.sample(n=sample_n, random_state=42)

        for _, _row in sampled.iterrows():
            # Naive
            rows.append({
                "app_id": int(app_id),
                "title": str(title),
                "rating": int(rating),
                "method": "naive",
                "prompt": make_naive_prompt(str(title), int(rating)),
                "generated_text": "",
            })

            # Engineered
            rows.append({
                "app_id": int(app_id),
                "title": str(title),
                "rating": int(rating),
                "method": "engineered",
                "prompt": make_engineered_prompt(str(title), int(rating)),
                "generated_text": "",
            })

    out_df = pd.DataFrame(rows)

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH} with shape: {out_df.shape}")
    print(out_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()