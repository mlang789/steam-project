from pathlib import Path
import joblib
import pandas as pd

from sbert_embedder import SBERTEmbedder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_PATH = "data/processed/reviews.csv"
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    df = pd.read_csv(DATA_PATH)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.len() >= 30].copy()

    X = df["review_text"]
    y = df["recommended"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("embed", SBERTEmbedder(MODEL_NAME)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    print("ROC-AUC:", auc)
    print(classification_report(y_test, pred, digits=4))

    joblib.dump(pipe, OUT_DIR / "baseline_model_sbert.joblib")
    print("Saved:", OUT_DIR / "baseline_model_sbert.joblib")

if __name__ == "__main__":
    main()
