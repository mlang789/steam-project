# genai/evaluate_diversity.py
"""
Evaluate diversity of generated Steam reviews.

Metrics (row-level):
- distinct_1: unique unigrams / total unigrams
- distinct_2: unique bigrams / total bigrams
- trigram_repeat_rate: repeated trigrams / total trigrams
- inter_review_sim_topk: mean cosine similarity to top-k nearest neighbors (global redundancy)

Embeddings for inter-review similarity:
- sbert (SentenceTransformers)
- tfidf (sklearn)

Inputs:
- --input CSV containing generated reviews
Required columns by default:
- generated_text (configurable via --text-col)
- method (optional but recommended for summary; configurable via --method-col)

Outputs (only if --save):
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


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")  # simple, robust tokenization


def tokenize(text: str, lower: bool = True) -> List[str]:
    s = str(text)
    tokens = WORD_RE.findall(s)
    if lower:
        tokens = [t.lower() for t in tokens]
    return tokens


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def distinct_n(tokens: List[str], n: int) -> float:
    ng = ngrams(tokens, n)
    return safe_ratio(len(set(ng)), len(ng))


def trigram_repeat_rate(tokens: List[str]) -> float:
    tri = ngrams(tokens, 3)
    if not tri:
        return 0.0
    from collections import Counter

    c = Counter(tri)
    repeats = sum(max(0, v - 1) for v in c.values())
    return safe_ratio(repeats, len(tri))


def embed_tfidf(texts: List[str], max_features: int = 50000):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts)  # sparse
    return X


def embed_sbert(texts: List[str], model_name: str, batch_size: int = 64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity friendly
    )
    return emb  # (n, d)


def inter_review_similarity_topk(reprs, topk: int = 5, metric: str = "cosine") -> np.ndarray:
    """
    For each item i: mean cosine similarity to its top-k nearest neighbors (excluding itself).
    Works for dense (SBERT) and sparse (TF-IDF) representations.
    """
    from sklearn.neighbors import NearestNeighbors

    if topk < 1:
        raise ValueError("--topk must be >= 1")

    k_query = topk + 1  # include self neighbor
    nn = NearestNeighbors(n_neighbors=k_query, metric=metric, algorithm="auto")
    nn.fit(reprs)

    distances, _ = nn.kneighbors(reprs, return_distance=True)
    sims = 1.0 - distances[:, 1:]  # drop self (distance 0)
    return sims.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--prefix", type=str, default="diversity_eval", help="Output prefix (used if --save).")
    parser.add_argument("--text-col", type=str, default="generated_text", help="Column containing generated reviews.")
    parser.add_argument("--method-col", type=str, default="method", help="Column for method names.")

    # Tokenization / lexical
    parser.add_argument("--lower", action="store_true", help="Lowercase tokens for lexical metrics.")
    parser.add_argument("--min-words", type=int, default=1, help="Drop rows with fewer words than this.")

    # Inter-review similarity
    parser.add_argument("--inter-sim", action="store_true", help="Compute inter-review similarity (global).")
    parser.add_argument("--embedder", choices=["sbert", "tfidf"], default="sbert", help="Embedding type.")
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="SBERT batch size.")
    parser.add_argument("--tfidf-max-features", type=int, default=50000, help="TF-IDF max features.")
    parser.add_argument("--topk", type=int, default=5, help="k nearest neighbors for inter-review similarity.")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, sample at most this many rows for inter-sim.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    # Saving
    parser.add_argument("--save", action="store_true", help="Save row-level and summary CSV files to reports/.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8", engine="python")

    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")

    if args.method_col not in df.columns:
        df[args.method_col] = "unknown"

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    used = df[df[args.text_col].str.strip().str.len() > 0].copy()

    # Lexical metrics
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
        raise ValueError("No rows left after filtering. Check --min-words / text column.")

    # Inter-review similarity (optional)
    used["inter_review_sim_topk"] = np.nan
    if args.inter_sim:
        sim_df = used
        if args.max_samples and len(sim_df) > args.max_samples:
            sim_df = sim_df.sample(n=args.max_samples, random_state=args.seed).copy()

        texts = sim_df[args.text_col].tolist()
        if args.embedder == "sbert":
            reprs = embed_sbert(texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)
        else:
            reprs = embed_tfidf(texts, max_features=args.tfidf_max_features)

        sims = inter_review_similarity_topk(reprs, topk=args.topk, metric="cosine")
        sim_df["inter_review_sim_topk"] = sims
        used.loc[sim_df.index, "inter_review_sim_topk"] = sim_df["inter_review_sim_topk"]

    # Summary by method
    agg_dict = dict(
        n=(args.text_col, "size"),
        mean_word_count=("word_count", "mean"),
        distinct1_mean=("distinct_1", "mean"),
        distinct2_mean=("distinct_2", "mean"),
        trigram_repeat_mean=("trigram_repeat_rate", "mean"),
    )

    if args.inter_sim:
        agg_dict.update(inter_sim_mean=("inter_review_sim_topk", "mean"))

    summary = (
        used.groupby(args.method_col, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("n", ascending=False)
    )

    print("=== Diversity evaluation summary (by method) ===")
    print(summary.to_string(index=False))

    # Save only if requested
    if args.save:
        out_rows_path = Path("reports") / f"{args.prefix}_div_rows.csv"
        out_summary_path = Path("reports") / f"{args.prefix}_div_summary.csv"
        out_rows_path.parent.mkdir(parents=True, exist_ok=True)

        used.to_csv(out_rows_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        summary.to_csv(out_summary_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

        print(f"\nSaved:\n- {out_rows_path}\n- {out_summary_path}")


if __name__ == "__main__":
    main()
