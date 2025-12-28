# Steam Reviews – Sentiment Analysis & Conditional Generation

## Project overview
This project studies Steam game reviews from two complementary perspectives:

1) **Sentiment prediction (baseline)**  
We show that predicting whether a review is recommended or not from its text is a relatively easy task using classical machine learning methods.

2) **Conditional text generation (main contribution)**  
We explore how a language model can generate Steam-like reviews conditioned on a game title and a target rating (derived from Steam’s binary label), and how generation quality and controllability change with:
- naive prompting
- prompt engineering
- **fine-tuning (LoRA) on Steam reviews**

The project follows a progressive methodology: starting from a simple baseline, then moving toward more original generative experiments.

---

## Data
We use review-level data collected from the official Steam Reviews API.

Each review includes:
- the review text
- a binary recommendation label (recommended / not recommended)
- metadata such as playtime at review time and vote counts
- the Steam app ID

Game titles are retrieved from the Steam Store API and merged with the review data.

**Rating proxy (coarse):** Steam reviews are binary, so we map:
- `recommended = 1` → `rating = 9`
- `recommended = 0` → `rating = 3`

This creates a simple numeric conditioning signal for generation (positive vs negative) while staying realistic (non-extreme).

Raw and processed datasets are not versioned in this repository.  
They are generated locally using the scripts provided in `src/`.

---

## Repository structure
steam-project/
├── src/            Reproducible scripts (data collection, cleaning, baseline)
├── notebooks/      EDA and modeling notebooks
├── genai/          Prompting + fine-tuning utilities
├── reports/        Saved metrics and evaluation outputs (generated locally)
├── data/           Generated locally (ignored by git)
├── README.md
├── requirements.txt
└── .gitignore

---

## Baseline: sentiment prediction
A simple baseline model is implemented to predict the review recommendation label from text:
- TF-IDF text representation (unigrams and bigrams)
- Logistic regression classifier

Despite its simplicity, this baseline achieves a strong ROC-AUC score, confirming that sentiment information is largely explicit in review text. The baseline is used both as a reference point and as a diagnostic tool for evaluating generated reviews (sentiment compliance).

---

## Conditional review generation
The main contribution of this project is the study of conditional review generation.

Given a game title and a coarse numeric rating derived from the recommendation label, we generate synthetic Steam-like user reviews using a language model.

We compare multiple strategies:

### 1) Naive prompting
A single-sentence instruction:
- *"Write a Steam user review for the game X with rating Y/10."*

### 2) Prompt engineering
A structured prompt with explicit constraints (length, tone, no spoilers, etc.).  
Prompt experiments and qualitative examples are documented in `genai/prompts.md`.

### 3) Fine-tuning (LoRA, Option A)
We fine-tune a small open-source model using **LoRA in 4-bit** (Google Colab GPU).

**Important design choice (Option A):** fine-tuning focuses on **Steam-like style + target sentiment** (via rating 3 vs 9) without enforcing strict format constraints during training (real Steam reviews do not reliably follow constraints such as exact word count or exact “2 pros / 1 con” structure).  
Those constraints are instead applied at inference time via prompt engineering and evaluated separately.

The fine-tuning dataset is built from real reviews as prompt → completion pairs:
- prompt: “Write a positive/negative Steam review for game X (rating Y/10)”
- completion: the real Steam review text

Dataset builder:
- `genai/make_dataset.py` → outputs `reports/sft_train.jsonl` and `reports/sft_val.jsonl`

---

## Evaluation
Generated reviews are evaluated using:
- **automatic sentiment compliance**, measured with a baseline classifier
- **simple structural checks** (e.g., word count constraints)
- qualitative analysis of realism, repetitiveness, and Steam-like style

### Automatic judges (TF-IDF vs SBERT)
We used two lightweight automatic sentiment judges:
- **TF-IDF + LogisticRegression** (bag-of-words baseline)
- **SBERT embeddings + LogisticRegression** (more robust to paraphrasing / style shifts)

On a balanced benchmark (25 positive prompts + 25 negative prompts), the fine-tuned model (`finetuned_v4`) reached:
- **Word-count compliance (100–140 words):** 0.72 (mean ~112 words)
- **Sentiment compliance (TF-IDF judge):** 0.54
- **Sentiment compliance (SBERT judge):** 0.60

This highlights both (i) the impact of decoding/format fixes on length control and (ii) the sensitivity of automatic evaluation: a bag-of-words judge can underestimate sentiment when generations use mixed phrasing, while embedding-based evaluation better captures overall meaning.

Evaluation script:
- `genai/evaluate_generations.py`  
Outputs (prefix-based):
- `reports/generation_eval_*_rows.csv`
- `reports/generation_eval_*_summary.csv`

---

## How to run the project

### Install dependencies
pip install -r requirements.txt

### Collect and clean data
python src/collect_reviews.py
python src/clean_reviews.py
python src/enrich_games.py
python src/build_reviews_with_title.py

### Train the baseline model (TF-IDF judge)
python src/train_baseline_textclf.py

### Build a prompt batch for generation
python genai/build_prompt_batch.py

Fill `generated_text` in `reports/prompt_batch.csv` manually (or via an external generation workflow).

### Evaluate generated reviews (TF-IDF or SBERT judge)
python genai/merge_for_eval.py
python genai/evaluate_generations.py --input reports/prompt_batch_all.csv --prefix generation_eval_v4
python genai/evaluate_generations.py --input reports/prompt_batch_all.csv --prefix generation_eval_v4_sbert

### Build fine-tuning dataset (SFT)
python genai/make_dataset.py

Fine-tuning is performed in Google Colab (GPU) using LoRA/4-bit on a small open-source model (e.g., TinyLlama 1.1B Chat). The notebook/workflow can be reproduced from the commands and scripts in this repository.

---

## Notes
- The goal of generation is stylistic and sentiment coherence rather than factual accuracy.
- Evaluation is intentionally lightweight and exploratory (automatic checks + qualitative inspection).
- Fine-tuning uses a pragmatic setup (LoRA + 4-bit quantization) suitable for student GPU environments (Colab).
