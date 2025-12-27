# genai/evaluate_generations.py
"""
Evaluate generated Steam reviews in reports/prompt_batch.csv using:
- a baseline sentiment classifier (TF-IDF + LogisticRegression)
- simple constraint checks (word length)

Inputs:
- reports/prompt_batch.csv (must contain: rating, method, generated_text)
- reports/baseline_model.joblib

Outputs:
- reports/generation_eval_rows.csv (row-level results)
- reports/generation_eval_summary.csv (method-level summary)
"""

from pathlib import Path

import pandas as pd
import joblib


PROMPT_BATCH_PATH = Path("reports/prompt_batch.csv")
MODEL_PATH = Path("reports/baseline_model.joblib")

OUT_ROWS_PATH = Path("reports/generation_eval_rows.csv")
OUT_SUMMARY_PATH = Path("reports/generation_eval_summary.csv")

# Engineered constraint in your prompts
MIN_WORDS = 100
MAX_WORDS = 140

# Your coarse rating mapping
POS_RATING_THRESHOLD = 7  # rating >= 7 => should be positive/recommended


def count_words(text: str) -> int:
    return len(str(text).strip().split())


def main() -> None:
    if not PROMPT_BATCH_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PROMPT_BATCH_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline model: {MODEL_PATH}. Run src/train_baseline_textclf.py and save it."
        )

    df = pd.read_csv(PROMPT_BATCH_PATH)

    required_cols = {"rating", "method", "generated_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"prompt_batch.csv missing columns: {sorted(missing)}")

    # Keep only rows where you actually filled a generation
    df["generated_text"] = df["generated_text"].fillna("").astype(str)
    used = df[df["generated_text"].str.strip().str.len() > 0].copy()

    if used.empty:
        raise ValueError("No filled generations found. Fill generated_text for some rows first.")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    proba_pos = model.predict_proba(used["generated_text"])[:, 1]
    pred_label = (proba_pos >= 0.5).astype(int)

    used["pred_proba_recommended"] = proba_pos
    used["pred_label"] = pred_label

    # Target label from rating
    used["target_label"] = (used["rating"].astype(float) >= POS_RATING_THRESHOLD).astype(int)

    # Compliance: predicted label matches target
    used["compliant"] = (used["pred_label"] == used["target_label"]).astype(int)

    # Word length constraints (mainly relevant for engineered prompts, but we compute for all)
    used["word_count"] = used["generated_text"].apply(count_words)
    used["length_ok_100_140"] = ((used["word_count"] >= MIN_WORDS) & (used["word_count"] <= MAX_WORDS)).astype(int)

    # Save row-level
    OUT_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    used.to_csv(OUT_ROWS_PATH, index=False, encoding="utf-8")

    # Summary by method
    summary = (
        used.groupby("method", dropna=False)
        .agg(
            n=("generated_text", "size"),
            compliance_rate=("compliant", "mean"),
            mean_pred_proba=("pred_proba_recommended", "mean"),
            length_ok_rate=("length_ok_100_140", "mean"),
            mean_word_count=("word_count", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8")

    # Print a readable summary
    print("=== Generation evaluation summary (by method) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved:\n- {OUT_ROWS_PATH}\n- {OUT_SUMMARY_PATH}")

    # Optional: show the filled rows quickly
    print("\n=== Filled rows preview ===")
    cols_preview = ["title", "rating", "method", "word_count", "pred_proba_recommended", "compliant"]
    cols_preview = [c for c in cols_preview if c in used.columns]
    print(used[cols_preview].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
