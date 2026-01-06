from pathlib import Path
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


# Paths

DATA_PATH = "data/processed/reviews.csv"
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data

df = pd.read_csv(DATA_PATH)

df["review_text"] = df["review_text"].fillna("").astype(str)
df = df[df["review_text"].str.len() >= 30].copy()


# Soft negation handling

def handle_negation_soft(text: str) -> str:
    text = re.sub(r"\bnot\s+(\w+)", r"not \1 NOT_\1", text)
    text = re.sub(r"\bno\s+(\w+)", r"no \1 NO_\1", text)
    text = re.sub(r"\bnever\s+(\w+)", r"never \1 NEVER_\1", text)
    return text

df["review_text"] = df["review_text"].apply(handle_negation_soft)

X = df["review_text"]
y = df["recommended"].astype(int)


# Pipeline: TF-IDF (bigram-focused) + Linear SVM

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=0.9,
        )
    ),
    (
        "clf",
        LinearSVC(class_weight="balanced")
    ),
])


# GridSearch (CV = validation)

param_grid = {
    "tfidf__min_df": [1, 2],
    "tfidf__max_features": [50000],
    "clf__C": [0.5, 1, 2],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
)


# Train on full dataset

grid.fit(X, y)

print("Best parameters:")
print(grid.best_params_)
print(f"Best CV score (macro-F1): {grid.best_score_:.4f}")

best_model = grid.best_estimator_


# Save final model

joblib.dump(best_model, OUTPUT_DIR / "svm_v2_model.joblib")

(OUTPUT_DIR / "svm_v2_metrics.txt").write_text(
    f"Best params:\n{grid.best_params_}\n\n"
    f"Best CV macro-F1: {grid.best_score_:.4f}\n",
    encoding="utf-8"
)

print("\nSaved reports/svm_v2.joblib")
print("Saved reports/svm_v2.txt")


# Inspect most discriminant features (global)

tfidf = best_model.named_steps["tfidf"]
clf = best_model.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]

top_pos = np.argsort(coefs)[-20:]
top_neg = np.argsort(coefs)[:20]

print("\nTop POSITIVE features:")
for i in top_pos[::-1]:
    print(feature_names[i], coefs[i])

print("\nTop NEGATIVE features:")
for i in top_neg:
    print(feature_names[i], coefs[i])
