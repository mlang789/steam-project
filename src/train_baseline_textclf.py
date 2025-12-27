from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/processed/reviews.csv"
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Basic cleaning
df["review_text"] = df["review_text"].fillna("").astype(str)
df = df[df["review_text"].str.len() >= 30].copy()

# Target as int
y = df["recommended"].astype(int)
X = df["review_text"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        n_jobs=None
    ))
])

model.fit(X_train, y_train)

import joblib
joblib.dump(model, OUTPUT_DIR / "baseline_model.joblib")
print("Saved reports/baseline_model.joblib")


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {auc:.4f}")
print(report)
print("Confusion matrix:\n", cm)

# Save metrics
(OUTPUT_DIR / "baseline_metrics.txt").write_text(
    f"ROC-AUC: {auc:.4f}\n\n{report}\n\nConfusion matrix:\n{cm}\n",
    encoding="utf-8"
)
print("Saved reports/baseline_metrics.txt")
