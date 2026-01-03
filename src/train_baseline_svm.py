from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/processed/reviews.csv"
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["review_text"] = df["review_text"].fillna("").astype(str)
df = df[df["review_text"].str.len() >= 30].copy()

X = df["review_text"]
y = df["recommended"].astype(int)

# ------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------
# Pipeline: TF-IDF + Linear SVM
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC(class_weight="balanced"))
])

# ------------------------------------------------------------------
# GridSearch: word vs char n-grams
# ------------------------------------------------------------------
param_grid = [
    {
        # WORD n-grams
        "tfidf__analyzer": ["word"],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_features": [50000],
        "clf__C": [0.1, 1, 10],
    },
    {
        # CHAR n-grams
        "tfidf__analyzer": ["char"],
        "tfidf__ngram_range": [(3, 5), (4, 6)],
        "tfidf__min_df": [3],
        "tfidf__max_features": [50000],
        "clf__C": [0.1, 1, 10],
    },
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
grid.fit(X_train, y_train)

print("Best parameters:")
print(grid.best_params_)
print(f"Best CV score (macro-F1): {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# ------------------------------------------------------------------
# Evaluation on test set
# ------------------------------------------------------------------
y_pred = best_model.predict(X_test)
scores = best_model.decision_function(X_test)

auc = roc_auc_score(y_test, scores)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest ROC-AUC: {auc:.4f}")
print(report)
print("Confusion matrix:\n", cm)

# ------------------------------------------------------------------
# Save model & metrics
# ------------------------------------------------------------------
import joblib
joblib.dump(best_model, OUTPUT_DIR / "svm_gridsearch_model.joblib")

(OUTPUT_DIR / "svm_gridsearch_metrics.txt").write_text(
    f"Best params:\n{grid.best_params_}\n\n"
    f"Best CV macro-F1: {grid.best_score_:.4f}\n\n"
    f"Test ROC-AUC: {auc:.4f}\n\n"
    f"{report}\n\nConfusion matrix:\n{cm}\n",
    encoding="utf-8"
)

print("\nSaved reports/svm_gridsearch_model.joblib")
print("Saved reports/svm_gridsearch_metrics.txt")

