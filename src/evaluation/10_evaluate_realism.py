# genai/evaluate_realism.py
"""
Évaluer le réalisme de reviews Steam générées à l’aide de SBERT + kNN.

Idée :
- On encode les reviews générées et un corpus de reviews réelles avec SBERT (embeddings normalisés).
- Pour chaque review générée, on cherche ses k plus proches voisins parmi les reviews réelles.
- On calcule ensuite :

Métriques (niveau review) :
- realism_sbert_topk : similarité cosinus moyenne avec les k voisins réels les plus proches
- nn_overlap : similarité cosinus maximale (voisin le plus proche, top-1)
- nn_overlap_flag : 1 si nn_overlap >= seuil, sinon 0

Résumé (par méthode) :
- realism_mean : moyenne de realism_sbert_topk
- nn_overlap_rate : proportion de nn_overlap_flag == 1 (potentiel "copier-coller" / mémorisation)
- nn_overlap_max : maximum de nn_overlap observé.

Exemple :
# 1. Naive
python src/evaluation/10_evaluate_realism.py --gen reports/prompt_batch_base_naive_clean.csv --real data/steam_real_reviews.csv --prefix "Approche Naive" --save

# 2. Engineered
python src/evaluation/10_evaluate_realism.py --gen reports/prompt_batch_base_engineered_clean.csv --real data/steam_real_reviews.csv --prefix "Approche Engineered" --save

# 3. Finetuned
python src/evaluation/10_evaluate_realism.py --gen reports/prompt_batch_finetuned_v5.csv --real data/steam_real_reviews.csv --prefix "Approche Finetuned" --save
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd


# Utilitaires

def word_count_simple(text: str) -> int:
    """Compte de mots basique (split sur les espaces)."""
    return len(str(text).strip().split())


# SBERT embeddings + kNN

def embed_sbert(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode une liste de textes avec SentenceTransformers.
    Les embeddings sont normalisés (L2), donc cosinus = produit scalaire.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def knn_realism_metrics(
    gen_emb: np.ndarray,
    real_emb: np.ndarray,
    topk: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pour chaque embedding généré :
    - realism_topk_mean : similarité cosinus moyenne avec les k voisins réels les plus proches
    - nn_overlap_max : similarité cosinus maximale (top-1)

    Hypothèse : embeddings déjà normalisés.
    """
    from sklearn.neighbors import NearestNeighbors

    if topk < 1:
        raise ValueError("--topk doit être >= 1")

    nn = NearestNeighbors(n_neighbors=topk, metric="cosine", algorithm="auto")
    nn.fit(real_emb)

    distances, _ = nn.kneighbors(gen_emb, return_distance=True)
    sims = 1.0 - distances  # distance cosinus -> similarité cosinus

    realism_topk_mean = sims.mean(axis=1)
    nn_overlap_max = sims[:, 0]  # top-1
    return realism_topk_mean, nn_overlap_max

# Programme principal

def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du réalisme (SBERT + kNN)")

    # Entrées
    parser.add_argument("--gen", type=str, required=True, help="CSV contenant les reviews générées.")
    parser.add_argument("--real", type=str, required=True, help="CSV contenant des reviews Steam réelles.")
    parser.add_argument("--gen-text-col", type=str, default="generated_text", help="Colonne texte (généré).")
    parser.add_argument("--real-text-col", type=str, default="review_text", help="Colonne texte (réel).")
    parser.add_argument("--method-col", type=str, default="method", help="Colonne indiquant la méthode.")
    parser.add_argument("--prefix", type=str, default="realism_eval", help="Préfixe des fichiers si --save.")

    # Filtrage
    parser.add_argument("--min-words", type=int, default=20, help="Supprime les reviews générées trop courtes.")

    # Paramètres SBERT
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2", help="Modèle SentenceTransformer.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="Batch size pour SBERT.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k voisins réels pour le score de réalisme.")
    parser.add_argument("--nn-threshold", type=float, default=0.9, help="Seuil de similarité pour overlap flag.")
    parser.add_argument(
        "--max-real",
        type=int,
        default=0,
        help="Si >0 : échantillonne au plus ce nombre de reviews réelles (accélère).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")

    # Sauvegarde
    parser.add_argument("--save", action="store_true", help="Sauvegarder les résultats dans reports/.")

    args = parser.parse_args()

    gen_path = Path(args.gen)
    real_path = Path(args.real)

    if not gen_path.exists():
        raise FileNotFoundError(f"Fichier introuvable (généré) : {gen_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"Fichier introuvable (réel) : {real_path}")

    gen_df = pd.read_csv(gen_path, encoding="utf-8", engine="python")
    real_df = pd.read_csv(real_path, encoding="utf-8", engine="python")

    # Vérification colonnes
    if args.gen_text_col not in gen_df.columns:
        raise ValueError(f"Colonne manquante dans gen CSV : '{args.gen_text_col}'. Colonnes: {list(gen_df.columns)}")
    if args.real_text_col not in real_df.columns:
        raise ValueError(f"Colonne manquante dans real CSV : '{args.real_text_col}'. Colonnes: {list(real_df.columns)}")

    # Si pas de colonne method, on en crée une
    if args.method_col not in gen_df.columns:
        gen_df[args.method_col] = "unknown"

    # Préparer les données générées
    
    gen_df[args.gen_text_col] = gen_df[args.gen_text_col].fillna("").astype(str)
    used = gen_df[gen_df[args.gen_text_col].str.strip().str.len() > 0].copy()

    used["word_count"] = used[args.gen_text_col].apply(word_count_simple)
    used = used[used["word_count"] >= args.min_words].copy()
    if used.empty:
        raise ValueError("Aucune review générée après filtrage. Vérifie --min-words et la colonne texte.")

    
    # Préparer les données réelles
    
    real_df[args.real_text_col] = real_df[args.real_text_col].fillna("").astype(str)
    real_used = real_df[real_df[args.real_text_col].str.strip().str.len() > 0].copy()
    if real_used.empty:
        raise ValueError("Aucune review réelle non vide trouvée.")

    if args.max_real and len(real_used) > args.max_real:
        real_used = real_used.sample(n=args.max_real, random_state=args.seed).copy()

    # Colonnes de sortie (on garde uniquement ce qui t’intéresse)
    used["realism_sbert_topk"] = np.nan
    used["nn_overlap"] = np.nan
    used["nn_overlap_flag"] = np.nan

    
    # Calcul SBERT + kNN
    
    gen_texts = used[args.gen_text_col].tolist()
    real_texts = real_used[args.real_text_col].tolist()

    gen_emb = embed_sbert(gen_texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)
    real_emb = embed_sbert(real_texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)

    realism_vals, nn_overlap_vals = knn_realism_metrics(gen_emb, real_emb, topk=args.topk)
    used["realism_sbert_topk"] = realism_vals
    used["nn_overlap"] = nn_overlap_vals
    used["nn_overlap_flag"] = (used["nn_overlap"] >= float(args.nn_threshold)).astype(int)

    
    # Résumé par méthode
    
    summary = (
        used.groupby(args.method_col, dropna=False)
        .agg(
            realism_mean=("realism_sbert_topk", "mean"),
            nn_overlap_rate=("nn_overlap_flag", "mean"),
            nn_overlap_max=("nn_overlap", "max"),
        )
        .reset_index()
    )

    print("=== Résumé évaluation réalisme (SBERT) — par méthode ===")
    print(summary.to_string(index=False))

    
    # Sauvegarde si demandé

    if args.save:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_md = out_dir / "results_realism.md"

        # On vérifie si le fichier existe déjà pour savoir si on "append" ('a') ou on crée ('w')
        file_exists = out_md.exists()
        mode = "a" if file_exists else "w"

        with open(out_md, mode, encoding="utf-8") as f:
            # Si le fichier est nouveau, on met le grand titre
            if not file_exists:
                f.write("# Résumé évaluation réalisme\n\n")
            
            # On ajoute un sous-titre basé sur le prefix ou la méthode pour s'y retrouver
            f.write(f"## Rapport : {args.prefix}\n\n")

            # On écrit le tableau
            try:
                # Nécessite 'pip install tabulate'
                f.write(summary.to_markdown(index=False))
            except ImportError:
                f.write(summary.to_string(index=False))
            
            f.write("\n\n") # Sauts de ligne pour aérer avant la prochaine exécution

        print(f"\nRésultats ajoutés à :\n- {out_md}")

if __name__ == "__main__":
    main()
