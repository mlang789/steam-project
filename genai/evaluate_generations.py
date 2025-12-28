# genai/evaluate_generations.py
"""
Evaluate generated Steam reviews using:
- a baseline sentiment classifier (TF-IDF + LogisticRegression)
- simple constraint checks (word length)

Inputs:
- --input CSV (must contain: rating, method, generated_text)
- reports/baseline_model.joblib

Outputs (controlled by --prefix):
- reports/<prefix>_rows.csv (row-level results)
- reports/<prefix>_summary.csv (method-level summary)
"""

from sbert_embedder import SBERTEmbedder  # needed for joblib unpickling

from pathlib import Path
import argparse

import pandas as pd
import joblib
import csv


MODEL_PATH = Path("reports/baseline_model_sbert.joblib")

# Engineered constraint in your prompts
MIN_WORDS = 100
MAX_WORDS = 140

# Your coarse rating mapping
POS_RATING_THRESHOLD = 7  # rating >= 7 => should be positive/recommended


def count_words(text: str) -> int:
    return len(str(text).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="reports/prompt_batch_all.csv",
        help="Input CSV containing prompts and generated_text.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generation_eval",
        help="Prefix for output files in reports/.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline model: {MODEL_PATH}. Run src/train_baseline_textclf.py first."
        )

    # engine="python" is safer when cells contain newlines
    df = pd.read_csv(input_path, encoding="utf-8", engine="python")

    required_cols = {"rating", "method", "generated_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

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

    # Word length constraints
    used["word_count"] = used["generated_text"].apply(count_words)
    used["length_ok_100_140"] = (
        (used["word_count"] >= MIN_WORDS) & (used["word_count"] <= MAX_WORDS)
    ).astype(int)

    out_rows_path = Path("reports") / f"{args.prefix}_rows.csv"
    out_summary_path = Path("reports") / f"{args.prefix}_summary.csv"

    out_rows_path.parent.mkdir(parents=True, exist_ok=True)
    used.to_csv(out_rows_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

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
        .sort_values("n", ascending=False)
    )
    summary.to_csv(out_summary_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print("=== Generation evaluation summary (by method) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved:\n- {out_rows_path}\n- {out_summary_path}")


if __name__ == "__main__":
    main()
