# Performance Analysis of Generative AI Models

This document presents a comprehensive evaluation of three approaches to Steam review generation (Naive, Engineered, Fine-tuned). The analysis examines factual quality, adherence to logical structures, semantic coherence, as well as the realism and diversity of generated texts.

## 1. Hallucination Analysis and Factual Quality

The first dimension of our evaluation focuses on the models' tendency to fabricate non-existent facts, including game mechanics, modes, and characters. This assessment was conducted through manual annotation by an external LLM judge, which systematically identified factual errors across 150+ generated reviews.

### Hallucination Rate Comparison

| Model | Game | Hallucination Rate | Global Average | Primary Error Typologies |
| :--- | :--- | :---: | :---: | :--- |
| **Naive** (Zero-Shot) | TF2 <br> CS2 | 56% <br> 60% | **58%** | **Genre confusion:** invention of "story" modes, RPG mechanics (levels, skill trees) and characters from third-party games. |
| **Engineered** (Instructions) | TF2 <br> CS2 | 68% <br> 62% | **65%** | **Increased fictional details:** invention of classes, developers and complex mechanics (grappling hooks, selfies). |
| **Fine-tuned** (Trained) | TF2 <br> CS2 | 28% <br> 34% | **31%** | **Technical residues:** metadata leakage, platform confusion or incorrect numbers, but overall genre respect. |

### Key Observations

**The Prompt Engineering Paradox (+7%)**: The "Engineered" model exhibits the highest error rate (65%). By forcing the model to be specific through detailed prompts, we inadvertently constrain it to generate details it does not master, triggering factually false over-creativity. This counterintuitive result demonstrates that explicit constraints can backfire when the model lacks domain knowledge.

**Fine-tuning Efficiency (-34%)**: Training on real data reduces the hallucination rate by more than half. The model integrates game-specific lexicon and boundaries, eliminating off-topic inventions such as open worlds in shooter games. This substantial improvement highlights the value of learning from examples rather than following explicit rules.

**Recurring Themes**: Untrained models tend to transform every game into a narrative RPG, regardless of actual genre. The fine-tuned model corrects this bias but occasionally suffers from data pollution, manifesting as identifiers or technical artifacts appearing in generated text.

---

## 2. Instruction Following and Structural Compliance

Building upon the factual quality analysis, we now examine the models' capacity to follow strict constraints. This evaluation focuses on two key aspects: argumentative structure (specifically, the "2 positive + 1 negative" constraint) and spoiler prohibition. These dimensions reveal how well models can execute quantitative instructions and domain-specific rules.

### Prompt Adherence Comparison

| Model | Game | Structure Failure (2+ / 1-) | "False Spoiler" Rate | Key Observation |
| :--- | :--- | :---: | :---: | :--- |
| **Naive** | Dota 2 <br> CS2 | N/A (Free prompt) | 14% <br> 22% | Invents scenarios through narrative reflex. |
| **Engineered** | Dota 2 <br> CS2 | **100%** <br> **96%** | 10% <br> 36% | **Near-total failure** on quantitative logic. |
| **Fine-tuned** | Dota 2 <br> CS2 | **90%** <br> **82%** | 4% <br> 10% | **Clear improvement** in structural understanding. |

### Analysis

**Limitations of Prompting Alone**: The explicit instruction "exactly 2 positive and 1 negative points" fails in 98% of cases with the Engineered model. Small-scale models (1.1B parameters) prioritize linguistic fluency over arithmetic rigor, treating numerical constraints as suggestions rather than hard requirements.

**Fine-tuning Contribution**: The success rate on structure improves from approximately 2% to 14%. While the score remains modest, the model demonstrates an attempt to balance its arguments, suggesting some internalization of structural patterns from the training data.

**Spoiler Management**: Paradoxically, the instruction "no spoilers" encouraged the Naive and Engineered models to invent stories to discuss. Fine-tuning corrects this flaw: having learned that games like CS2 lack narrative campaigns, the narrative hallucination rate drops drastically (from 36% to 4% on CS2). This demonstrates that domain knowledge acquisition through training outperforms explicit prohibition instructions.

---

## 3. Semantic Coherence: Rating Versus Generated Text

Having assessed factual accuracy and structural compliance, we now turn to semantic alignment. This section analyzes the correspondence between the target rating (input) and the sentiment expressed in the generated text. Misalignment in this dimension indicates the model's inability to maintain consistent emotional tone throughout the review.

### Conformity Rate Between Target Rating and Sentiment

| Evaluated Game | Target Rating | Expected Sentiment | Success Rate | Nature of Divergences |
| :--- | :---: | :---: | :---: | :--- |
| **Dota 2** | 9 / 10 | POSITIVE | **100%** | None. |
| **Dota 2** | 3 / 10 | NEGATIVE | **80%** | Internal inconsistency (e.g., "surprised by the quality" for 3/10). |
| **Team Fortress 2** | 9 / 10 | POSITIVE | **100%** | None. |
| **Team Fortress 2** | 3 / 10 | NEGATIVE | **84%** | Affective conflict (e.g., "my favorite game" for 3/10). |
| **Counter-Strike 2** | 3 / 10 | NEGATIVE | **76%** | Hallucinated satisfaction. |
| **AVERAGE** | - | - | **88.0%** | **Predominant positivity bias.** |

### Interpretation

A compliance bias is evident in the results. The model excels at generating positive reviews (100% success) but exhibits resistance to critique. Approximately 20% of negative reviews contain contradictory praise, creating internal tension within the text.

This pattern suggests that the base model's initial alignment (TinyLlama) favors a helpful and enthusiastic tone, creating cognitive dissonance when generating harsh criticism. The model appears to "struggle" with negativity, occasionally inserting positive remarks even when instructed to write a critical review. This behavior underscores a fundamental limitation of zero-shot and prompt-based approaches when the desired output contradicts the model's learned preferences.

---

## 4. Realism and Diversity Analysis

Transitioning from internal consistency to external validity, we now examine how generated reviews compare to authentic human-written content. This dual analysis addresses two complementary questions: do the reviews resemble real Steam reviews (realism), and do they exhibit sufficient variation to avoid repetitiveness (diversity)? These metrics were computed using SBERT embeddings and n-gram statistics.

### Realism Synthesis (Semantic Proximity)

**Naive and Engineered (Score approximately 0.69)**: These approaches produce original texts that do not copy the existing corpus. No cases of plagiarism were detected. The moderate similarity score indicates that while the content is novel, it maintains reasonable proximity to authentic reviews in semantic space.

**Fine-tuned (Score approximately 0.70)**: Fine-tuning offers the highest semantic realism. A minimal trace of memorization (0.02% overlap) is detected, which is negligible but indicates the model is beginning to retain typical phrases from the training dataset. This slight memorization does not constitute plagiarism but rather reflects successful pattern learning.

### Diversity Synthesis

**Lexical Diversity (Internal Metrics)**: The Naive and Engineered models use more varied vocabulary locally, exhibiting a higher count of unique words per review. Fine-tuning slightly reduces this lexical variety, as the model converges toward the vocabulary distribution observed in training data.

**Semantic Diversity (Global Metrics)**: Despite their lexical richness, prompt-based models (Naive/Engineered) exhibit high inter-review similarity (approximately 0.84). They tend to repeat the same argumentative structures and talking points across all games, resulting in monotonous content at scale. In contrast, the Fine-tuned model shows lower similarity (approximately 0.77), generating reviews that are more semantically distinct from one another.

### Diversity Conclusion

This reveals a critical paradox: fine-tuning produces individually simpler texts but achieves greater global variation and less repetitiveness than prompt engineering approaches. The reduction in lexical complexity does not translate to reduced diversity; instead, it reflects more authentic, game-specific language use. Prompt-based approaches, while demonstrating surface-level vocabulary variation, fail to generate truly distinct argumentative content across different contexts.

---

## 5. Automated Evaluation Reliability: LLM-as-a-Judge Validation

To complete our methodological validation, we assessed the reliability of our automated SBERT sentiment classifier by comparing its predictions against those of an expert external LLM judge on a stratified 300-review sample. This validation ensures that our scalable automated metrics accurately reflect human-level judgment.

### SBERT Judge Performance Against LLM Expert

| Data Segment | Accuracy | F1-Score (Positive) | F1-Score (Negative) | Reliability |
| :--- | :---: | :---: | :---: | :--- |
| **Global** | **87.67%** | **0.92** | **0.69** | **High** |
| **Subset: Naive** | 93.00% | 0.96 | 0.81 | Very High |
| **Subset: Engineered** | 84.00% | 0.89 | 0.62 | Moderate |
| **Subset: Fine-tuned** | 86.00% | 0.91 | 0.65 | High |

### Evaluation Tool Validity

The SBERT judge demonstrates robust performance for global evaluation, achieving nearly 88% agreement with expert LLM annotations. However, a class imbalance is notable: the classifier performs significantly better at detecting positive reviews (F1: 0.92) than negative ones (F1: 0.69). This disparity stems from the greater linguistic complexity of negative reviews, which often employ nuance, sarcasm, or mixed sentiments.

Accuracy decreases slightly on complex texts (Engineered/Fine-tuned approaches) compared to the stereotypical outputs of the Naive model. This pattern suggests that more sophisticated generation strategies produce reviews that are harder to classify automatically, potentially due to their closer resemblance to authentic human writing with its inherent ambiguities.

Despite these limitations, the high overall accuracy validates our dual evaluation strategy: using automated SBERT classification for scalable sentiment assessment while maintaining rigorous quality control through LLM-as-a-Judge verification on critical dimensions such as hallucinations and structural compliance.