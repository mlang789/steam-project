# genai/evaluate_diversity.py
"""
Évaluer la diversité de reviews Steam générées.

Métriques calculées :
- word_count : nombre de mots (tokenisation simple)
- distinct_1 : nb de unigrammes uniques / nb total de unigrammes
- distinct_2 : nb de bigrammes uniques / nb total de bigrammes
- trigram_repeat_rate : nb de trigrammes répétés / nb total de trigrammes
- inter_review_sim_topk : similarité moyenne (cosinus) avec les k voisins les plus proches
  parmi les autres reviews générées (redondance globale).

Entrée :
- --input : CSV contenant les reviews générées
Colonnes attendues par défaut :
- generated_text (modifiable via --text-col)
- method (optionnel mais recommandé, modifiable via --method-col)

Sorties (uniquement si --save) :
- reports/<prefix>_div_rows.csv
- reports/<prefix>_div_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")  # tokenisation simple et robuste



# Tokenisation + n-grams

def tokenize(text: str, lower: bool = True) -> List[str]:
    """Découpe un texte en tokens (mots) avec une regex simple."""
    tokens = WORD_RE.findall(str(text))
    if lower:
        tokens = [t.lower() for t in tokens]
    return tokens


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Construit la liste des n-grams sous forme de tuples."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def safe_ratio(num: int, den: int) -> float:
    """Division sécurisée (évite division par zéro)."""
    return float(num) / float(den) if den > 0 else 0.0


def distinct_n(tokens: List[str], n: int) -> float:
    """Distinct-n = nb n-grams uniques / nb n-grams total."""
    ng = ngrams(tokens, n)
    return safe_ratio(len(set(ng)), len(ng))


def trigram_repeat_rate(tokens: List[str]) -> float:
    """
    Taux de répétition des trigrammes :
    (nb de trigrammes en excès) / (nb total de trigrammes)
    Exemple : si un trigramme apparaît 3 fois, on compte 2 répétitions.
    """
    tri = ngrams(tokens, 3)
    if not tri:
        return 0.0

    from collections import Counter

    c = Counter(tri)
    repeats = sum(max(0, v - 1) for v in c.values())
    return safe_ratio(repeats, len(tri))



# SBERT embeddings + similarité inter-reviews

def embed_sbert(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """
    Encode les textes avec SentenceTransformers.
    Les embeddings sont normalisés (L2), donc la similarité cosinus est directe.
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


def inter_review_similarity_topk(embeddings: np.ndarray, topk: int = 5) -> np.ndarray:
    """
    Pour chaque review i : similarité cosinus moyenne avec ses top-k voisins les plus proches,
    en excluant elle-même (voisin de distance 0).
    """
    from sklearn.neighbors import NearestNeighbors

    if topk < 1:
        raise ValueError("--topk doit être >= 1")

    k_query = topk + 1  # inclut soi-même
    nn = NearestNeighbors(n_neighbors=k_query, metric="cosine", algorithm="auto")
    nn.fit(embeddings)

    distances, _ = nn.kneighbors(embeddings, return_distance=True)
    sims = 1.0 - distances[:, 1:]  # on enlève le voisin "soi-même"
    return sims.mean(axis=1)



# Programme principal

def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation de la diversité (lexicale + redondance globale).")

    parser.add_argument("--input", type=str, required=True, help="Chemin du CSV d’entrée.")
    parser.add_argument("--prefix", type=str, default="diversity_eval", help="Préfixe (si --save).")
    parser.add_argument("--text-col", type=str, default="generated_text", help="Colonne contenant le texte généré.")
    parser.add_argument("--method-col", type=str, default="method", help="Colonne indiquant la méthode.")

    # Tokenisation / métriques lexicales
    parser.add_argument("--lower", action="store_true", help="Mettre les tokens en minuscules.")
    parser.add_argument("--min-words", type=int, default=1, help="Supprime les reviews trop courtes.")

    # Similarité inter-reviews (SBERT uniquement)
    parser.add_argument("--inter-sim", action="store_true", help="Calcule la similarité inter-reviews.")
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2", help="Modèle SentenceTransformer.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="Batch size SBERT.")
    parser.add_argument("--topk", type=int, default=5, help="k voisins pour la similarité inter-reviews.")
    parser.add_argument("--max-samples", type=int, default=0, help="Si >0, échantillonne pour accélérer.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")

    # Sauvegarde
    parser.add_argument("--save", action="store_true", help="Sauvegarde les CSV dans reports/.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8", engine="python")

    if args.text_col not in df.columns:
        raise ValueError(f"Colonne texte manquante '{args.text_col}'. Disponibles : {list(df.columns)}")

    if args.method_col not in df.columns:
        df[args.method_col] = "unknown"

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    used = df[df[args.text_col].str.strip().str.len() > 0].copy()

    
    # Métriques lexicales
    
    word_counts: List[int] = []
    d1_list: List[float] = []
    d2_list: List[float] = []
    tri_rep_list: List[float] = []

    for text in used[args.text_col].tolist():
        toks = tokenize(text, lower=args.lower)
        word_counts.append(len(toks))
        d1_list.append(distinct_n(toks, 1))
        d2_list.append(distinct_n(toks, 2))
        tri_rep_list.append(trigram_repeat_rate(toks))

    used["word_count"] = word_counts
    used["distinct_1"] = d1_list
    used["distinct_2"] = d2_list
    used["trigram_repeat_rate"] = tri_rep_list

    used = used[used["word_count"] >= args.min_words].copy()
    if used.empty:
        raise ValueError("Aucune ligne après filtrage. Vérifie --min-words et la colonne texte.")

    
    # Similarité inter-reviews (optionnel)
    
    used["inter_review_sim_topk"] = np.nan
    if args.inter_sim:
        sim_df = used
        if args.max_samples and len(sim_df) > args.max_samples:
            sim_df = sim_df.sample(n=args.max_samples, random_state=args.seed).copy()

        texts = sim_df[args.text_col].tolist()
        emb = embed_sbert(texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)
        sims = inter_review_similarity_topk(emb, topk=args.topk)

        sim_df["inter_review_sim_topk"] = sims
        used.loc[sim_df.index, "inter_review_sim_topk"] = sim_df["inter_review_sim_topk"]

    
    # Résumé par méthode (uniquement les métriques utiles)
    
    agg_dict = dict(
        n=(args.text_col, "size"),
        mean_word_count=("word_count", "mean"),
        distinct1_mean=("distinct_1", "mean"),
        distinct2_mean=("distinct_2", "mean"),
        trigram_repeat_mean=("trigram_repeat_rate", "mean"),
        inter_sim_mean=("inter_review_sim_topk", "mean"),
    )

    summary = (
        used.groupby(args.method_col, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("n", ascending=False)
    )

    print("=== Résumé évaluation diversité (par méthode) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
