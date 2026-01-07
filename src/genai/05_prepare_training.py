import sys
from pathlib import Path
# ajout du chemin racine pour trouver src.config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import FILES, GENAI_INPUTS_DIR

# on utilise le dataset final généré par l'étape 02
DATA_PATH = FILES["train"]["final"]

# on sauvegarde dans le dossier propre
TRAIN_PATH = GENAI_INPUTS_DIR / "sft_train.jsonl"
VAL_PATH = GENAI_INPUTS_DIR / "sft_val.jsonl"

# définitions des constantes (notes et limites de caractères)
POS_RATING = 9
NEG_RATING = 3
ALLOWED_RATINGS = {POS_RATING, NEG_RATING}

MIN_REVIEW_CHARS = 80
MAX_REVIEW_CHARS = 1200

def make_sft_prompt(title: str, rating: int) -> str:
    # construction du prompt instruction pour le fine-tuning
    sentiment = "positive" if rating >= 7 else "negative"
    return "\n".join([
        "You are a regular Steam user.",
        "",
        f'Write a {sentiment} Steam user review for the game "{title}".',
        f"Target rating: {rating}/10.",
        "",
        "Guidelines:",
        "- Casual, honest tone",
        "- No spoilers",
        "- Output only the review text",
        "",
        "Review:",
    ])

def write_jsonl(path: Path, rows: list[dict]) -> None:
    # écriture du fichier au format jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    # vérification de l'existence du fichier source
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    # chargement et validation des colonnes requises
    df = pd.read_csv(DATA_PATH)
    required = {"title", "rating", "review_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # conversion des types et gestion des valeurs nulles
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["rating"] = df["rating"].astype(int)

    # filtrage : on garde seulement les notes cibles et les textes de bonne longueur
    df = df[df["rating"].isin(ALLOWED_RATINGS)].copy()
    df = df[df["review_text"].str.len() >= MIN_REVIEW_CHARS].copy()
    df = df[df["review_text"].str.len() <= MAX_REVIEW_CHARS].copy()
    df = df[df["title"].str.strip().str.len() > 0].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    # construction des paires prompt / completion
    rows = []
    for _, r in df.iterrows():
        prompt = make_sft_prompt(r["title"], int(r["rating"]))
        completion = r["review_text"].strip()
        rows.append({"prompt": prompt, "completion": completion})

    # division du jeu de données (90% train, 10% validation)
    train_rows, val_rows = train_test_split(rows, test_size=0.1, random_state=42)

    # sauvegarde sur le disque
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)

    print(f"Saved:\n- {TRAIN_PATH} ({len(train_rows)} rows)\n- {VAL_PATH} ({len(val_rows)} rows)")

if __name__ == "__main__":
    main()