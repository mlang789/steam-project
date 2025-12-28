# genai/merge_for_eval.py
"""
Merge prompt batches (naive/engineered + finetuned) into a single CSV for evaluation.

Inputs:
- reports/prompt_batch.csv
- reports/prompt_batch_finetuned_balanced_50.csv

Output:
- reports/prompt_batch_all.csv
"""

from pathlib import Path
import pandas as pd


BASE_PATH = Path("reports/prompt_batch.csv")
FINETUNED_PATH = Path("reports/prompt_batch_finetuned_balanced_50_v4.csv")
OUT_PATH = Path("reports/prompt_batch_all.csv")

REQUIRED_COLS = ["app_id", "title", "rating", "method", "prompt", "generated_text"]


def read_csv_robust(path: Path) -> pd.DataFrame:
    # engine="python" is more robust when cells contain newlines.
    return pd.read_csv(path, encoding="utf-8", engine="python")


def ensure_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}")
    return df[REQUIRED_COLS].copy()


def main() -> None:
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASE_PATH}")
    if not FINETUNED_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FINETUNED_PATH}")

    base = read_csv_robust(BASE_PATH)
    finetuned = read_csv_robust(FINETUNED_PATH)

    base = ensure_columns(base, "prompt_batch.csv")
    finetuned = ensure_columns(finetuned, "prompt_batch_finetuned_balanced_50.csv")

    # Normalize types
    for df in (base, finetuned):
        df["app_id"] = df["app_id"].astype(int)
        df["rating"] = df["rating"].astype(int)
        df["title"] = df["title"].fillna("").astype(str)
        df["method"] = df["method"].fillna("").astype(str)
        df["prompt"] = df["prompt"].fillna("").astype(str)
        df["generated_text"] = df["generated_text"].fillna("").astype(str)

    all_df = pd.concat([base, finetuned], ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Saved: {OUT_PATH} | shape={all_df.shape}")
    print("\nCounts by method:")
    print(all_df["method"].value_counts(dropna=False).to_string())

    filled = (all_df["generated_text"].str.strip().str.len() > 0).sum()
    print(f"\nFilled generations: {filled}/{len(all_df)}")
    print("Note: evaluation will only use rows where generated_text is non-empty.")


if __name__ == "__main__":
    main()
