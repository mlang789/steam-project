import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sentence_transformers import SentenceTransformer
from src.config import FILES, MODELS_DIR, EVAL_DIR
from src.utils import SBERTEmbedder

class SBERTEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(list(X), show_progress_bar=True)


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Entraîner un modèle Juge (SBERT ou TF-IDF)")
    parser.add_argument("--model", type=str, choices=["sbert", "tfidf"], default="sbert")
    args = parser.parse_args()

    # --- Chargement du dataset TRAIN ---
    df_train = pd.read_csv(FILES["train"]["final"])
    df_train["review_text"] = df_train["review_text"].fillna("").astype(str)
    X_train_full = df_train["review_text"]
    y_train_full = (df_train["rating"] > 5).astype(int)

    # --- Définition du modèle ---
    base_clf = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)

    if args.model == "sbert":
        steps = [
            ("embed", SBERTEmbedder()),
            ("clf", base_clf)
        ]
    else:
        steps = [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
            ("clf", base_clf)
        ]

    pipe = Pipeline(steps)

    # --- Entraînement sur tout le train ---
    print("Entraînement sur tout le dataset TRAIN...")
    pipe.fit(X_train_full, y_train_full)

    # --- Chargement du dataset VALIDATION finale ---
    df_val = pd.read_csv(FILES["validation"]["final"])
    df_val["review_text"] = df_val["review_text"].fillna("").astype(str)
    X_val = df_val["review_text"]
    y_val = (df_val["rating"] > 5).astype(int)

    # --- Évaluation ---
    # Seuil implicite = 0
    if args.model == "sbert":
        scores = pipe.named_steps['clf'].decision_function(pipe.named_steps['embed'].transform(X_val))
    else:
        scores = pipe.named_steps['clf'].decision_function(pipe.named_steps['tfidf'].transform(X_val))

    y_pred_val = (scores >= 0).astype(int)
    auc = roc_auc_score(y_val, scores)
    report = classification_report(y_val, y_pred_val, digits=4)

    print(f"\n--- Évaluation sur VALIDATION finale ---")
    print(report)
    print(f"ROC-AUC Score : {auc:.4f}")

    # --- Sauvegarde des métriques ---
    metrics_path = EVAL_DIR / f"metrics_{args.model}.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Threshold: 0 (implicit)\n")
        f.write(f"ROC-AUC: {auc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    print(f"Métriques sauvegardées dans : {metrics_path}")

    # --- Sauvegarde du modèle ---
    out_path = MODELS_DIR / f"judge_model_{args.model}.joblib"
    joblib.dump(pipe, out_path)
    print(f"Modèle sauvegardé sous : {out_path}")


if __name__ == "__main__":
    main()