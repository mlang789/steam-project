# FILE: src/config.py

from pathlib import Path

# --- Configuration des Jeux ---
# (Tes listes TRAIN_APP_IDS et VALIDATION_APP_IDS restent inchangées ici)
TRAIN_APP_IDS = [570, 730, 440, 1091500, 1245620, 1517290, 1599340, 2357570, 1272080, 1547000]
VALIDATION_APP_IDS = [548430, 892970, 1097840, 1940340]

# --- Chemins de base ---
# Astuce : On remonte d'un cran car config.py est maintenant dans src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_OUTPUTS_DIR = DATA_DIR / "outputs_gen_ai_models" # On garde ça

# --- Organisation de Reports (Outputs) ---
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = REPORTS_DIR / "models"           # Pour les .joblib
GENAI_INPUTS_DIR = REPORTS_DIR / "genai_inputs" # Pour les JSONL et Prompts
EVAL_DIR = REPORTS_DIR / "evaluation"         # Pour les résultats

# Création automatique des dossiers
for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_OUTPUTS_DIR, MODELS_DIR, GENAI_INPUTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Dictionnaire de fichiers (Mapping) ---
# Changement ici : 'games' devient 'titles'
FILES = {
    "train": {
        "raw_reviews": RAW_DIR / "reviews_raw_train.csv",
        "titles": PROCESSED_DIR / "titles_train.csv",    # RENOMMÉ
        "final": PROCESSED_DIR / "dataset_train.csv"
    },
    "validation": {
        "raw_reviews": RAW_DIR / "reviews_raw_val.csv",
        "titles": PROCESSED_DIR / "titles_val.csv",      # RENOMMÉ
        "final": PROCESSED_DIR / "dataset_val.csv"
    }
}