from pathlib import Path

# --- Configuration des Jeux ---
# Jeux pour l'ENTRAÎNEMENT (Train + Test split)
TRAIN_APP_IDS = [
    570,      # Dota 2
    730,      # Counter-Strike 2
    440,      # Team Fortress 2
    1091500,  # Cyberpunk 2077
    1245620,  # Elden Ring
    1517290,  # Battlefield 2042
    1599340,  # The Day Before
    2357570,  # Overwatch 2
    1272080,  # Payday 3
    1547000,  # GTA Trilogy Definitive Edition
]

# Jeux pour la validation (Totalement invisibles au modèle pendant l'entraînement)
VALIDATION_APP_IDS = [
    548430,   # Deep Rock Galactic
    892970,   # Valheim
    1097840,  # Halo Infinite
    1940340,  # Redfall
]

# --- Chemins de fichiers ---
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")

# Création automatique des dossiers
for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dictionnaire pour mapper les noms de fichiers selon le mode
FILES = {
    "train": {
        "raw_reviews": RAW_DIR / "reviews_raw_train.csv",
        "games": PROCESSED_DIR / "games_train.csv",
        "final": PROCESSED_DIR / "dataset_train.csv"
    },
    "validation": {
        "raw_reviews": RAW_DIR / "reviews_raw_val.csv",
        "games": PROCESSED_DIR / "games_val.csv",
        "final": PROCESSED_DIR / "dataset_val.csv"
    }
}