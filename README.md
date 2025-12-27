# Steam Reviews – Sentiment Analysis & Conditional Generation

## Project overview
This project studies Steam game reviews from two complementary perspectives:

1) Sentiment prediction (baseline)  
We show that predicting whether a review is recommended or not from its text is a relatively easy task using classical machine learning methods.

2) Conditional text generation (main contribution)  
We explore how a language model can generate Steam-like reviews conditioned on a game title and a target rating, and how the quality and controllability of generation depend on prompt design (prompt engineering).

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

Raw and processed datasets are not versioned in this repository.  
They are generated locally using the scripts provided in `src/`.

---

## Repository structure
steam-project/
├── src/            Reproducible scripts (data collection, cleaning, baseline)
├── notebooks/      EDA and modeling notebooks
├── genai/          Prompt-based generation experiments
├── reports/        Saved metrics and evaluation outputs
├── data/           Generated locally (ignored by git)
├── README.md
├── requirements.txt
└── .gitignore

---

## Baseline: sentiment prediction
A simple baseline model is implemented to predict the review recommendation label from text:
- TF-IDF text representation (unigrams and bigrams)
- Logistic regression classifier

Despite its simplicity, this baseline achieves a strong ROC-AUC score, confirming that sentiment information is largely explicit in review text. The baseline is used both as a reference point and as a diagnostic tool for evaluating generated reviews.

---

## Conditional review generation
The main contribution of this project is the study of conditional review generation.

Given a game title and a coarse numeric rating derived from the recommendation label, we generate synthetic Steam-like user reviews using a large language model.

We compare two prompting strategies:
- naive prompting
- prompt engineering with structured constraints (length, tone, balance between positive and negative points)

Prompt experiments and qualitative examples are documented in `genai/prompts.md`.

---

## Evaluation
Generated reviews are evaluated using a combination of:
- automatic evaluation, based on sentiment compliance measured with the baseline classifier
- simple structural checks (word count constraints)
- qualitative analysis of realism and style

Due to the generative nature of the task, evaluation is intentionally lightweight and exploratory.

---

## How to run the project

Install dependencies:
pip install -r requirements.txt

Collect and clean data:
python src/collect_reviews.py
python src/clean_reviews.py

Train the baseline model:
python src/train_baseline_textclf.py

Build generation prompts:
python genai/build_prompt_batch.py

Evaluate generated reviews:
python genai/evaluate_generations.py

---

## Notes
- The goal of generation is stylistic and sentiment coherence rather than factual accuracy.
- The project focuses on methodological comparison rather than application deployment.
- Fine-tuning of a language model is considered as potential future work.
