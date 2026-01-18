# Sentiment Analysis and Conditional Generation of Steam Reviews

## Context

Steam is the world's largest digital video game distribution platform with over 120 million active users. The platform hosts millions of user-written reviews ranging from single sentences to detailed essays. These reviews represent authentic, unscripted opinions about games, making Steam an exceptional data source for NLP research. Unlike curated datasets, Steam reviews exhibit diverse writing styles, varying expertise levels, sarcasm, and nuanced sentiment that closely mirror real-world text generation challenges. This project leverages the Steam API to collect authentic gaming reviews as a foundation for studying how AI systems can learn to imitate community-specific writing patterns while maintaining factual accuracy.

## Overview

This project demonstrates a comprehensive pipeline for evaluating and generating high-quality synthetic text at scale. We built and deployed three key machine learning systems: a **SBERT-based sentiment classifier** trained for real-time semantic analysis, a **LoRA-fine-tuned LLM** optimized for domain-specific review generation, and an **LLM-as-a-Judge evaluation framework** capable of assessing factual accuracy, structural compliance, and semantic quality across thousands of generated documents.

The research investigates a challenge in generative AI: how to maintain factual integrity and stylistic authenticity when generating domain-specific content under varying constraint regimes. By comparing zero-shot prompting, detailed prompt engineering, and parameter-efficient fine-tuning on TinyLlama 1.1B, we demonstrate that explicit constraints can paradoxically degrade output quality, while targeted training outperforms instruction-following by 34 percentage points in factual accuracy.

**Key Technical Achievements:**
- Implemented dual-validation evaluation: SBERT classifier (87.67% agreement with expert LLM) + manual LLM-as-a-Judge annotation on 300+ reviews
- Developed automated hallucination detection, structure compliance checking, and spoiler identification systems
- Achieved 31% hallucination rate through fine-tuning versus 65% with detailed prompts (paradoxically worse)
- Created scalable n-gram and embedding-based diversity metrics for evaluating semantic repetitiveness across 20K+ generated reviews
- Built end-to-end pipeline from Steam API data collection to cloud GPU training (Google Colab) to multi-dimensional evaluation

## Repository Structure

The file organization clearly separates source code, data, and analyses.

```text
steam-project/
├── USAGE.md                    # Comprehensive execution guide for all scripts
├── analyse.md                  # Detailed analysis report of generation results
├── notebooks/                  # Jupyter notebooks (exploration) and Google Colab (GPU fine-tuning)
├── data/                       # Local data (ignored by git)
├── reports/                    # Saved models, prompts, and evaluation results
├── src/                        # Modular Python source code
│   ├── data/                   # Collection and cleaning scripts
│   ├── judge/                  # Classifier training
│   ├── genai/                  # Generation pipeline and fine-tuning preparation
│   └── evaluation/             # Quantitative analysis scripts
│       └── EVALUATION_GUIDE.md # Detailed evaluation suite documentation
└── requirements.txt            # Project dependencies
```

**Key Documentation Files:**
- **[USAGE.md](./USAGE.md)**: Complete pipeline execution guide with all commands
- **[analyse.md](./analyse.md)**: In-depth performance analysis with hallucination rates, diversity metrics, and LLM-as-a-Judge validation
- **[src/evaluation/EVALUATION_GUIDE.md](./src/evaluation/EVALUATION_GUIDE.md)**: Detailed usage guide for the three evaluation scripts (quality, diversity, realism)

The notebooks required to reproduce GPU training (Google Colab) or visualize data are located in the `notebooks/` folder.

## Quick Start

### Installation

The project requires **Python 3.10+**. Install dependencies:

```bash
# Clone the repository
git clone https://github.com/mlang789/steam-project.git
cd steam-project

# Install dependencies
pip install -r requirements.txt
```

### Rapid Execution (End-to-End)

```bash
# 1. Collect data
python src/data/01_collect.py train && python src/data/01_collect.py validation

# 2. Process data
python src/data/02_process.py train && python src/data/02_process.py validation

# 3. Train sentiment judge
python src/judge/03_train.py --model sbert

# 4. Generate prompts
python src/genai/04_generate_prompts.py
python src/genai/05_prepare_training.py

# 5. Run inference (use Colab notebooks for GPU acceleration)
# Then merge results:
python src/genai/06_merge_outputs.py

# 6. Evaluate
python src/genai/07_evaluate.py --model sbert
```

## Data

### Dataset Characteristics

The data comes from the **Steam public API** with the following properties:

| Dataset Split | Games | Reviews per Game | Total Reviews | Source |
|---------------|-------|------------------|---------------|--------|
| **Training** | 10 | ~2,000 | ~20,000 | Dota 2, CS2, TF2, Cyberpunk, Elden Ring, etc. |
| **Validation** | 4 | ~2,000 | ~8,000 | Deep Rock Galactic, Cuphead, Baldur's Gate 3, Palworld |

### Rating Transformation

For conditional generation, we transform binary recommendations into numeric ratings:
- **Recommended** → 9/10 (highly positive sentiment)
- **Not Recommended** → 3/10 (negative but argumentative)

This transformation provides nuanced conditioning while avoiding extreme ratings (0 or 10) that might bias generation.

**Note**: Raw data is not stored in the repository. Use `src/data/` scripts to reconstruct datasets locally from the Steam API.

## Methodology and Approaches

We compare three generation paradigms to produce synthetic reviews:

### 1. Naive Approach (Zero-Shot)
- **Strategy**: Minimal prompt ("Write a Steam review for X rated Y/10")
- **Model**: TinyLlama 1.1B base model
- **Pros**: Fast, baseline comparison
- **Cons**: 58% hallucination rate, genre confusion

### 2. Prompt Engineering (Structured Instructions)
- **Strategy**: Complex constraints (word count, "2 positive + 1 negative" structure, no spoilers)
- **Model**: Same base model with detailed instructions
- **Pros**: Better formatting control
- **Cons**: **65% hallucinations** (paradoxically worse!), 98% failure on quantitative constraints

### 3. Fine-Tuning (LoRA Supervised Learning)
- **Strategy**: Parameter-efficient fine-tuning on ~5,000 real Steam reviews
- **Model**: TinyLlama 1.1B with LoRA adapters
- **Training**: Google Colab with GPU acceleration
- **Pros**: **31% hallucinations** (best), learns gaming vocabulary and conventions
- **Cons**: Requires GPU resources, occasional metadata leakage

## Pipeline Usage

The complete project execution consists of four successive stages.

### 1. Collection and Preparation
These commands download raw data and create cleaned CSV files.

```bash
python src/data/01_collect.py train
python src/data/01_collect.py validation
python src/data/02_process.py train
python src/data/02_process.py validation
```

### 2. Judge Training
We create a model capable of automatically evaluating whether a generated review is positive or negative. This serves as a control metric.

```bash
python src/judge/03_train.py --model sbert
```

### 3. Text Generation
This phase prepares instruction files.
*   Generate prompts for approaches without training:
    ```bash
    python src/genai/04_generate_prompts.py
    ```
*   Prepare data for fine-tuning:
    ```bash
We measure multiple quality dimensions:
- **Hallucination detection**: Factual accuracy via manual annotation and Steam API validation
- **Structure compliance**: Adherence to prompt constraints (word count, argumentation structure)
- **Sentiment coherence**: Judge model predictions vs. target ratings
- **Diversity metrics**: Distinct-n, trigram repetition, inter-review similarity
- **Realism scoring**: k-NN similarity to real reviews using SBERT embeddings

For the complete evaluation workflow, see [USAGE.md](./USAGE.md).

Quick evaluation example:
```bash
# Evaluate sentiment alignment
python src/genai/07_evaluate.py --model sbert

# Analyze hallucinations, structure, and spoilers
python src/evaluation/08_evaluate_quality.py

# Measure diversity and realism
python src/evaluation/09_evaluate_diversity.py
python src/evaluation/10_evaluate_realism.py
```
    ```

Once these files are generated, the notebooks in `notebooks/` must be used (for example on Colab) to perform GPU inference and training. The results must then be merged:

```bash
python src/genai/06_merge_outputs.py
```

### 4. Evaluation

We employ a **multi-method evaluation strategy** combining automated and LLM-based assessments:

- **LLM-as-a-Judge**: External language model evaluates:
  - Factual hallucinations (invented game mechanics, modes, characters)
  - Structural compliance ("2 positive + 1 negative" constraint)
  - Spoiler detection (narrative reveals)
  
- **Automated Metrics**:
  - **Sentiment coherence**: SBERT judge model predictions vs. target ratings
  - **Diversity metrics**: Distinct-n, trigram repetition, inter-review similarity
  - **Realism scoring**: k-NN similarity to real reviews using SBERT embeddings
  
- **Validation**: SBERT judge validated against expert LLM on 300-review sample (87.67% agreement)

For the complete evaluation workflow, see [USAGE.md](./USAGE.md). For detailed evaluation script usage, see [src/evaluation/EVALUATION_GUIDE.md](./src/evaluation/EVALUATION_GUIDE.md).

**Quick evaluation examples:**
```bash
# Evaluate sentiment alignment
python src/genai/07_evaluate.py --model sbert

# Analyze hallucinations, structure, and spoilers (generates prompts for LLM judge)
python src/evaluation/08_evaluate_quality.py all

# Measure diversity
python src/evaluation/09_evaluate_diversity.py \
  --input reports/genai_inputs/prompt_batch_filled.csv --inter-sim --save

# Measure realism
python src/evaluation/10_evaluate_realism.py \
  --gen reports/genai_inputs/prompt_batch_filled.csv \
  --real data/raw/reviews_raw_train.csv --save
```

## Key Results

### 1. Hallucination Analysis (Factual Quality)

| Approach | Avg. Hallucination Rate | Improvement | Key Errors |
|----------|------------------------|-------------|------------|
| **Naive** | 58% | Baseline | Genre confusion (invents RPG mechanics in FPS games) |
| **Engineered** | **65%** | **+7% worse** | Forced specificity creates fictional details |
| **Fine-tuned** | **31%** | **-34% better** | Minor metadata leaks, respects game genres |

**Critical Discovery**: Detailed prompt constraints force models to invent plausible-sounding but factually incorrect details when lacking domain knowledge.

### 2. Instruction Following & Structure Compliance

| Metric | Naive | Engineered | Fine-tuned |
|--------|-------|------------|------------|
| **Structure compliance** ("2+/1-" rule) | N/A | **2%** (98% fail) | **14%** (7× better) |
| **Spoiler prevention** (CS2) | 22% fail | 36% fail | **10%** fail |
| **Sentiment-rating alignment** | 71% | 75% | **86%** |

**Finding**: Small models (1.1B params) prioritize linguistic fluency over arithmetic precision. Fine-tuning improves structural understanding but cannot fully enforce quantitative constraints.

### 3. Realism & Diversity

| Metric | Naive/Engineered | Fine-tuned | Winner |
|--------|------------------|------------|--------|
| **Semantic realism** (similarity to real reviews) | 0.69 | **0.70** | Fine-tuned |
| **Lexical diversity** (unique words/review) | Higher | Lower | Naive/Eng |
| **Global semantic diversity** (inter-review similarity) | 0.84 (repetitive) | **0.77** | Fine-tuned |

**Paradox**: Prompt-based approaches use more varied vocabulary locally but produce globally repetitive arguments. Fine-tuned models generate simpler but more distinct reviews.

### 4. LLM-as-a-Judge Validation

We employed a **dual evaluation strategy**: automated SBERT classification for scalability, validated against expert LLM judgments.

**SBERT Judge vs. LLM-as-a-Judge** (300-review validation sample):
- **Overall accuracy**: 87.67%
- **F1-Score (Positive)**: 0.92 (excellent)
- **F1-Score (Negative)**: 0.69 (moderate - struggles with sarcasm/nuance)

**LLM Judge Tasks**:
- Hallucination annotation for 150+ reviews (TF2, CS2, Dota 2)
- Structure compliance verification ("2+/1-" rule)
- Spoiler identification across all three approaches

This hybrid approach enables **scalable automated metrics** while maintaining **rigorous quality control** through LLM verification.

### Summary: Why Fine-Tuning Wins

| Dimension | Winner | Margin |
|-----------|--------|--------|
| Factual accuracy | Fine-tuning | -34% hallucinations |
| Sentiment coherence | Fine-tuning | +15% alignment |
| Semantic diversity | Fine-tuning | Better variation |
| Structure following | Fine-tuning | 7× improvement |
| **Overall** | **Fine-tuning** | **Decisive across all metrics** |

**For detailed analysis**, see [analyse.md](./analyse.md) with full statistical breakdowns, error taxonomies, and example outputs.

## Technical Stack

- **Data Collection**: Steam Web API, `requests`, `pandas`
- **ML Classification**: `scikit-learn` (LinearSVC), `sentence-transformers` (SBERT)
- **Generative AI**: Hugging Face Transformers, LoRA fine-tuning (PEFT)
- **Evaluation**: 
  - **LLM-as-a-Judge**: External LLM for hallucination detection, structure compliance, and spoiler assessment
  - **SBERT Classifier**: Automated sentiment prediction (validated at 87.67% accuracy)
  - **Embedding-based Metrics**: k-NN similarity for realism, cosine distance for diversity
- **Infrastructure**: Python 3.10+, Jupyter/Google Colab for GPU tasks

## Academic Contributions

1. **Empirical demonstration** that prompt engineering can degrade factual quality when forcing specificity
2. **Quantification** of fine-tuning benefits across multiple dimensions (hallucinations -34%, coherence +15%)
3. **Discovery** that local lexical richness ≠ global semantic diversity
4. **Validation** of LLM-as-a-Judge methodology: automated SBERT evaluation achieves 87.67% agreement with expert LLM annotations, enabling scalable quality assessment
5. **Comprehensive evaluation framework** combining LLM judgment, embedding-based metrics, and n-gram analysis for multi-dimensional quality assessment