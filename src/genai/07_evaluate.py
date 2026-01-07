import sys
from pathlib import Path
import argparse
import pandas as pd
import joblib
import csv
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.config import MODELS_DIR, EVAL_DIR, GENAI_INPUTS_DIR

from src.utils import SBERTEmbedder

MIN_WORDS = 100
MAX_WORDS = 140
POS_RATING_THRESHOLD = 7

def count_words(text: str) -> int:
    return len(str(text).strip().split())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(GENAI_INPUTS_DIR / "prompt_batch_filled.csv"),
        help="Input CSV containing prompts and generated_text.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generation_eval",
        help="Prefix for output files in reports/results/.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["sbert", "tfidf"],
        default="sbert",
        help="Which judge model to use? (Default: sbert)"
    )
    args = parser.parse_args()

    # construction du chemin du modèle
    model_filename = f"judge_model_{args.model}.joblib"
    model_path = MODELS_DIR / model_filename

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing judge model: {model_path}. Run 'python src/judge/03_train_judge.py --model {args.model}' first."
        )

    # chargement des données
    df = pd.read_csv(input_path, encoding="utf-8", engine="python")
    
    required_cols = {"rating", "method", "generated_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df["generated_text"] = df["generated_text"].fillna("").astype(str)
    used = df[df["generated_text"].str.strip().str.len() > 0].copy()

    if used.empty:
        raise ValueError("No filled generations found. Fill generated_text for some rows first.")

    # chargement du modèle juge
    print(f"Loading judge model ({args.model.upper()}) from {model_path}...")
    model = joblib.load(model_path)

    # prédiction
    # le pipeline gère lui-même la transformation (sbert ou tf-idf)
    try:
        # cas probabiliste (rare avec linearsvc sauf si calibré)
        proba_pos = model.predict_proba(used["generated_text"])[:, 1]
        pred_label = (proba_pos >= 0.5).astype(int)
    except AttributeError:
        # cas linearsvc standard (fonction de décision)
        scores = model.decision_function(used["generated_text"])
        proba_pos = 1 / (1 + np.exp(-scores)) # sigmoïde pour simuler une probabilité
        pred_label = (scores >= 0).astype(int)

    used["pred_proba_recommended"] = proba_pos
    used["pred_label"] = pred_label

    # calcul des métriques
    used["target_label"] = (used["rating"].astype(float) >= POS_RATING_THRESHOLD).astype(int)
    used["compliant"] = (used["pred_label"] == used["target_label"]).astype(int)
    used["word_count"] = used["generated_text"].apply(count_words)
    used["length_ok_100_140"] = (
        (used["word_count"] >= MIN_WORDS) & (used["word_count"] <= MAX_WORDS)
    ).astype(int)

    # sauvegarde
    # on ajoute le nom du modèle au préfixe pour ne pas écraser les résultats
    final_prefix = f"{args.prefix}_{args.model}"
    
    out_rows_path = EVAL_DIR / f"{final_prefix}_rows.csv"
    out_summary_path = EVAL_DIR / f"{final_prefix}_summary.csv"
    out_txt_path = EVAL_DIR / f"{final_prefix}_report.txt"

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

    summary_str = summary.to_string(index=False)
    
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=== GENERATION EVALUATION REPORT ===\n\n")
        f.write(f"Model Type:  {args.model.upper()}\n")
        f.write(f"Input File:  {input_path}\n")
        f.write(f"Model Path:  {model_path}\n")
        f.write("-" * 40 + "\n\n")
        f.write(summary_str)
        f.write("\n\n" + "-" * 40 + "\n")
        f.write("Legend:\n")
        f.write("- compliance_rate: % of reviews matching the target sentiment (Pos/Neg)\n")
    
    print("\n" + "="*40)
    print(f"      RÉSULTATS ({args.model.upper()})")
    print("="*40)
    print(summary_str)
    print("="*40)
    print(f"\nFichiers générés dans {EVAL_DIR} :")
    print(f"1. {out_rows_path.name}")
    print(f"2. {out_summary_path.name}")
    print(f"3. {out_txt_path.name}")


if __name__ == "__main__":
    main()