import argparse
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer
from config import FILES, REPORTS_DIR

class SBERTEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(list(X), show_progress_bar=True)

def find_best_threshold(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * (p * r) / (p + r + 1e-10)
    best_idx = np.argmax(f1)
    return t[best_idx]

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Entraîner un modèle Juge (SBERT ou TF-IDF)")
    parser.add_argument("--model", type=str, choices=["sbert", "tfidf"], default="sbert",
                        help="Type de modèle à entraîner")
    args = parser.parse_args()

    # --- Chargement du dataset TRAIN ---
    df_train = pd.read_csv(FILES["train"]["final"])
    df_train["review_text"] = df_train["review_text"].fillna("").astype(str)
    X_train_full = df_train["review_text"]
    y_train_full = (df_train["rating"] > 5).astype(int)
    groups = df_train["app_id"]

    # --- Split interne 80/20 par jeux pour seuil ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X_train_full, y_train_full, groups=groups))
    X_train, y_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
    X_val_split, y_val_split = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

    # --- Définition du modèle ---
    base_clf = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)

    if args.model == "sbert":
        steps = [
            ("embed", SBERTEmbedder()),
            ("clf", clf)
        ]
    else:
        steps = [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
            ("clf", clf)
        ]

    pipe = Pipeline(steps)

    print("Entraînement sur 80% des jeux pour optimiser le seuil F1...")
    pipe.fit(X_train, y_train)

    # --- Optimisation du seuil sur la mini-validation interne ---
    y_proba_split = pipe.predict_proba(X_val_split)[:, 1]
    threshold = find_best_threshold(y_val_split, y_proba_split)
    print(f"Seuil optimal trouvé sur split interne : {threshold:.3f}")

    # --- Ré-entraînement sur tout le train ---
    print("Ré-entraînement sur tout le dataset TRAIN...")
    pipe.fit(X_train_full, y_train_full)

    # --- Chargement du dataset VALIDATION finale ---
    df_val = pd.read_csv(FILES["validation"]["final"])
    df_val["review_text"] = df_val["review_text"].fillna("").astype(str)
    X_val = df_val["review_text"]
    y_val = (df_val["rating"] > 5).astype(int)

    # --- Évaluation sur validation finale ---
    y_proba_val = pipe.predict_proba(X_val)[:, 1]
    y_pred_val = (y_proba_val >= threshold).astype(int)

    auc = roc_auc_score(y_val, y_proba_val)
    report = classification_report(y_val, y_pred_val, digits=4)

    print(f"\n--- Évaluation sur VALIDATION finale ---")
    print(f"Seuil utilisé : {threshold:.3f}")
    print(report)
    print(f"ROC-AUC Score : {auc:.4f}")

    # Sauvegarde des métriques
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / f"metrics_{args.model}.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Threshold: {threshold:.6f}\n")
        f.write(f"ROC-AUC: {auc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    print(f"Métriques sauvegardées dans : {metrics_path}")


    # --- Sauvegarde du modèle + seuil ---
    pipe.named_steps['clf'].threshold_ = threshold
    out_path = REPORTS_DIR / f"judge_model_{args.model}.joblib"
    joblib.dump(pipe, out_path)
    print(f"Modèle sauvegardé sous : {out_path}")

if __name__ == "__main__":
    main()