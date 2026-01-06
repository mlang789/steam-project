import sys
from pathlib import Path
# Ajout du chemin racine pour trouver src.config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import pandas as pd
from src.config import FILES

def main(mode):
    paths = FILES[mode]
    print(f"--- Construction du dataset : {mode.upper()} ---")

    if not paths["raw_reviews"].exists() or not paths["titles"].exists():
        print("Fichiers manquants. Lancez d'abord 01_collect_data.py")
        return

    # 1. Load & Merge
    reviews = pd.read_csv(paths["raw_reviews"])
    titles = pd.read_csv(paths["titles"])
    
    df = reviews.merge(titles, on="app_id", how="left")
    
    # 2. Clean
    initial_len = len(df)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.len() >= 30] # Trop court
    df = df.drop_duplicates(subset=["app_id", "review_text"]) # Doublons
    
    print(f"Nettoyage : {initial_len} -> {len(df)} reviews.")

    # 3. Format Labels
    # Mapping: Recommended (1) -> 9/10, Not Rec (0) -> 3/10
    df["rating"] = df["recommended"].astype(int).map({1: 9, 0: 3})
    
    # Save
    cols = ["app_id", "title", "rating", "review_text"] # On garde l'essentiel
    df[cols].to_csv(paths["final"], index=False)
    print(f"✅ Dataset final généré : {paths['final']}")
    print(df.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "validation"])
    args = parser.parse_args()
    main(args.mode)