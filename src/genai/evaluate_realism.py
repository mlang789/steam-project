# genai/evaluate_realism.py
"""
Evaluate realism of generated Steam reviews.

Metrics (row-level):
1) Perplexity (optional):
   - ppl: exp(token-level cross-entropy) under a causal LM (proxy for fluency)

2) SBERT realism (optional):
   - realism_sbert_topk: mean cosine similarity to top-k nearest REAL reviews
   - nn_overlap: max cosine similarity to any REAL review (top-1)
   - nn_overlap_flag: 1 if nn_overlap >= threshold else 0

Summary (grouped by method):
- ppl_mean, ppl_median (if --perplexity)
- realism_mean, nn_overlap_rate, nn_overlap_max (if --sbert)

Nothing is saved by default; use --save to write to reports/.

Example:
python .\\genai\\evaluate_realism.py `
  --gen .\\reports\\prompt_batch_base_engineered_clean.csv `
  --real .\\data\\steam_real_reviews.csv `
  --gen-text-col generated_text `
  --real-text-col review_text `
  --perplexity --ppl-model distilgpt2 --ppl-batch-size 8 `
  --sbert --sbert-model all-MiniLM-L6-v2 --topk 5 --nn-threshold 0.9 `
  --save --prefix engineered_realism
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def word_count_simple(text: str) -> int:
    return len(str(text).strip().split())


# ----------------------------
# Perplexity (Causal LM)
# ----------------------------
def compute_perplexities_hf(
    texts: List[str],
    model_name: str = "distilgpt2",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "auto",
) -> np.ndarray:
    """
    Compute per-example perplexity using a HuggingFace causal LM.

    We compute per-token negative log-likelihood for each example (masking padding),
    then ppl = exp(mean_nll_per_example).

    Returns: np.ndarray of shape (len(texts),)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # GPT2-like models often have no pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    ppl_out: List[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Build labels and ignore pad tokens
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Mask out ignored labels
            valid_mask = shift_labels != -100

            # log softmax then gather log prob of true token
            log_probs = torch.log_softmax(shift_logits, dim=-1)

            # clamp index to avoid gather issues where label is -100
            gather_index = shift_labels.clamp(min=0).unsqueeze(-1)
            token_logp = log_probs.gather(dim=-1, index=gather_index).squeeze(-1)

            # Negative log-likelihood, masked
            nll = -token_logp * valid_mask

            nll_sum = nll.sum(dim=1)
            tok_count = valid_mask.sum(dim=1).clamp(min=1)

            loss_per_sample = (nll_sum / tok_count).detach().cpu().numpy()
            ppl_batch = np.exp(loss_per_sample)
            ppl_out.extend(ppl_batch.tolist())

    return np.array(ppl_out, dtype=float)


# ----------------------------
# SBERT embeddings + kNN
# ----------------------------
def embed_sbert(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts with SentenceTransformers; embeddings are L2-normalized.
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
    For each generated embedding:
    - realism_topk_mean: mean cosine similarity to top-k nearest REAL embeddings
    - nn_overlap_max: max cosine similarity (top-1)

    Assumes embeddings are normalized.
    """
    from sklearn.neighbors import NearestNeighbors

    if topk < 1:
        raise ValueError("--topk must be >= 1")

    nn = NearestNeighbors(n_neighbors=topk, metric="cosine", algorithm="auto")
    nn.fit(real_emb)

    distances, _ = nn.kneighbors(gen_emb, return_distance=True)
    sims = 1.0 - distances  # cosine distance -> cosine similarity

    realism_topk_mean = sims.mean(axis=1)
    nn_overlap_max = sims[:, 0]
    return realism_topk_mean, nn_overlap_max


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--gen", type=str, required=True, help="CSV with generated reviews.")
    parser.add_argument("--real", type=str, required=True, help="CSV with REAL Steam reviews.")
    parser.add_argument("--gen-text-col", type=str, default="generated_text", help="Column in gen CSV.")
    parser.add_argument("--real-text-col", type=str, default="review_text", help="Column in real CSV.")
    parser.add_argument("--method-col", type=str, default="method", help="Method column in gen CSV.")
    parser.add_argument("--prefix", type=str, default="realism_eval", help="Output prefix if --save.")

    # Filtering
    parser.add_argument("--min-words", type=int, default=20, help="Drop generated rows with fewer words.")

    # What to compute
    parser.add_argument("--perplexity", action="store_true", help="Compute perplexity (fluency).")
    parser.add_argument("--sbert", action="store_true", help="Compute SBERT realism + NN overlap.")

    # Perplexity params
    parser.add_argument("--ppl-model", type=str, default="distilgpt2", help="HF causal LM name.")
    parser.add_argument("--ppl-batch-size", type=int, default=8, help="Batch size for perplexity.")
    parser.add_argument("--ppl-max-length", type=int, default=512, help="Max tokens per example.")
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")

    # SBERT params
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="Batch size for SBERT encode.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k REAL neighbors for realism score.")
    parser.add_argument("--nn-threshold", type=float, default=0.9, help="Threshold for overlap flag.")
    parser.add_argument(
        "--max-real",
        type=int,
        default=0,
        help="If >0, sample at most this many REAL reviews (speed).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Saving
    parser.add_argument("--save", action="store_true", help="Save results to reports/.")

    args = parser.parse_args()

    gen_path = Path(args.gen)
    real_path = Path(args.real)

    if not gen_path.exists():
        raise FileNotFoundError(f"Missing generated CSV: {gen_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"Missing real CSV: {real_path}")

    gen_df = pd.read_csv(gen_path, encoding="utf-8", engine="python")
    real_df = pd.read_csv(real_path, encoding="utf-8", engine="python")

    if args.gen_text_col not in gen_df.columns:
        raise ValueError(f"gen CSV missing '{args.gen_text_col}'. Columns: {list(gen_df.columns)}")
    if args.real_text_col not in real_df.columns:
        raise ValueError(f"real CSV missing '{args.real_text_col}'. Columns: {list(real_df.columns)}")

    if args.method_col not in gen_df.columns:
        gen_df[args.method_col] = "unknown"

    # Prepare generated
    gen_df[args.gen_text_col] = gen_df[args.gen_text_col].fillna("").astype(str)
    used = gen_df[gen_df[args.gen_text_col].str.strip().str.len() > 0].copy()

    used["word_count"] = used[args.gen_text_col].apply(word_count_simple)
    used = used[used["word_count"] >= args.min_words].copy()
    if used.empty:
        raise ValueError("No generated rows left after filtering. Check --min-words / text column.")

    # Prepare real
    real_df[args.real_text_col] = real_df[args.real_text_col].fillna("").astype(str)
    real_used = real_df[real_df[args.real_text_col].str.strip().str.len() > 0].copy()
    if real_used.empty:
        raise ValueError("No non-empty real reviews found.")

    if args.max_real and len(real_used) > args.max_real:
        real_used = real_used.sample(n=args.max_real, random_state=args.seed).copy()

    # Require at least one metric group
    if not args.perplexity and not args.sbert:
        raise ValueError("Select at least one metric group: --perplexity and/or --sbert.")

    # Init columns
    used["perplexity"] = np.nan
    used["realism_sbert_topk"] = np.nan
    used["nn_overlap"] = np.nan
    used["nn_overlap_flag"] = np.nan

    # Compute perplexity
    if args.perplexity:
        ppl = compute_perplexities_hf(
            used[args.gen_text_col].tolist(),
            model_name=args.ppl_model,
            batch_size=args.ppl_batch_size,
            max_length=args.ppl_max_length,
            device=args.device,
        )
        used["perplexity"] = ppl

    # Compute SBERT realism + overlap
    if args.sbert:
        gen_texts = used[args.gen_text_col].tolist()
        real_texts = real_used[args.real_text_col].tolist()

        gen_emb = embed_sbert(gen_texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)
        real_emb = embed_sbert(real_texts, model_name=args.sbert_model, batch_size=args.sbert_batch_size)

        realism_mean, nn_overlap = knn_realism_metrics(gen_emb, real_emb, topk=args.topk)
        used["realism_sbert_topk"] = realism_mean
        used["nn_overlap"] = nn_overlap
        used["nn_overlap_flag"] = (used["nn_overlap"] >= float(args.nn_threshold)).astype(int)

    # Summary by method
    agg = {}
    if args.perplexity:
        agg.update(
            ppl_mean=("perplexity", "mean"),
            ppl_median=("perplexity", "median"),
        )
    if args.sbert:
        agg.update(
            realism_mean=("realism_sbert_topk", "mean"),
            nn_overlap_rate=("nn_overlap_flag", "mean"),
            nn_overlap_max=("nn_overlap", "max"),
        )

    summary = (
        used.groupby(args.method_col, dropna=False)
        .agg(**agg)
        .reset_index()
    )

    print("=== Realism evaluation summary (by method) ===")
    print(summary.to_string(index=False))

    # Save if requested
    if args.save:
        out_rows = Path("reports") / f"{args.prefix}_real_rows.csv"
        out_sum = Path("reports") / f"{args.prefix}_real_summary.csv"
        out_rows.parent.mkdir(parents=True, exist_ok=True)

        used.to_csv(out_rows, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        summary.to_csv(out_sum, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

        print(f"\nSaved:\n- {out_rows}\n- {out_sum}")


if __name__ == "__main__":
    main()
