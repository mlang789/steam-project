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

    # Vérification de l'existence des fichiers générés à l'étape précédente
    if not paths["raw_reviews"].exists() or not paths["titles"].exists():
        print("Fichiers manquants. Lancez d'abord 01_collect_data.py")
        return

    # 1. Chargement et fusion
    reviews = pd.read_csv(paths["raw_reviews"])
    titles = pd.read_csv(paths["titles"])
    
    # Association du titre du jeu à chaque review
    df = reviews.merge(titles, on="app_id", how="left")
    
    # 2. Nettoyage
    initial_len = len(df)
    # Gestion des valeurs nulles et conversion en chaîne de caractères
    df["review_text"] = df["review_text"].fillna("").astype(str)
    
    # Suppression des reviews trop courtes (< 30 caractères)
    df = df[df["review_text"].str.len() >= 30] 
    
    # Suppression des doublons (même jeu, même texte)
    df = df.drop_duplicates(subset=["app_id", "review_text"]) 
    
    print(f"Nettoyage : {initial_len} -> {len(df)} reviews.")

    # 3. Formatage des labels
    # Conversion du vote binaire : Recommandé (1) -> 9, Non recommandé (0) -> 3
    df["rating"] = df["recommended"].astype(int).map({1: 9, 0: 3})
    
    # Sauvegarde
    cols = ["app_id", "title", "rating", "review_text"] # Sélection des colonnes pertinentes
    df[cols].to_csv(paths["final"], index=False)
    print(f"Dataset final généré : {paths['final']}")
    print(df.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "validation"])
    args = parser.parse_args()
    main(args.mode)