# Steam Reviews – Sentiment Analysis & Conditional Generation

## Project overview
This project studies Steam game reviews from two complementary perspectives:

1) Sentiment prediction (baseline)  
We show that predicting whether a review is recommended or not from its text is a relatively easy task using classical machine learning methods.

2) Conditional text generation (main contribution)  
We explore how a language model can generate Steam-like reviews conditioned on a target sentiment, and how the quality of generation depends on prompt design (prompt engineering) and fine-tuning on domain-specific data.

The project follows a progressive methodology: starting from a simple baseline, then moving toward more original generative experiments.

---

## Data
We use review-level data collected from the official Steam Reviews API.

Each review includes:
- the review text
- a binary recommendation label (recommended / not recommended)
- metadata such as playtime at review time and vote counts

Raw and processed datasets are not versioned in this repository.  
They are generated locally using the scripts provided in `src/`.

---

## Repository structure
steam-project/
├── src/            Reproducible scripts (data collection, cleaning, baseline)
├── notebooks/      EDA and modeling notebooks
├── genai/          Prompt experiments and fine-tuning preparation
├── reports/        Saved metrics and figures
├── data/           Generated locally (ignored by git)
├── README.md
├── requirements.txt
└── .gitignore

---

## Baseline: sentiment prediction
A simple baseline model is implemented to predict the review recommendation label from text:
- TF-IDF text representation (unigrams and bigrams)
- Logistic regression classifier

Despite its simplicity, this baseline achieves a strong ROC-AUC score, confirming that sentiment information is largely explicit in review text. This motivates shifting the focus toward generative modeling rather than further classifier optimization.

---

## Generative modeling
The main contribution of this project is the study of conditional review generation.

We compare:
- naive prompting
- prompt engineering with structured constraints
- fine-tuning of a small language model on Steam reviews

Prompt experiments and qualitative comparisons are documented in `genai/prompts.md`.

---

## How to run the project

Install dependencies:
pip install -r requirements.txt

Collect and clean data:
python src/collect_reviews.py
python src/clean_reviews.py

Train the baseline model:
python src/train_baseline_textclf.py

---

## Notes
- The goal of generation is stylistic and sentiment coherence rather than factual accuracy.
- The project focuses on methodological comparison rather than application deployment.
- Possible extensions include richer conditioning signals and multilingual modeling.
