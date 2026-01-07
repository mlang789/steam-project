# 🛡️ Guide d'Utilisation : Suite d'Évaluation GenAI

Ce dossier contient trois scripts Python situés dans `src/evaluation/` permettant d'évaluer la qualité, la diversité et le réalisme des reviews générées par vos modèles.

## 📋 Pré-requis Communs

Tous les scripts nécessitent un environnement Python avec les bibliothèques suivantes :

```bash
pip install pandas numpy requests tqdm scikit-learn sentence-transformers tabulate
```

> **Note :** `sentence-transformers` est lourd mais nécessaire pour les analyses sémantiques (SBERT).

---

## 1. Évaluation de la Qualité & Jugement LLM

**Script :** `08_evaluate_quality.py`

Ce script génère des prompts pour qu'un "Juge IA" (GPT-4/Claude) détecte les hallucinations, les spoilers ou les erreurs de structure. Il permet aussi de calculer une métrique de précision de sentiment (SBERT vs Juge).

**Syntaxe :**
```bash
python src/evaluation/08_evaluate_quality.py [TACHE]
```

### Tâches Disponibles

| Argument | Description | Sorties (dans `evaluation/`) |
| :--- | :--- | :--- |
| `all` | Exécute toutes les générations de prompts + préparation SBERT. | Tous les fichiers ci-dessous. |
| `hallucination` | Prompts pour détecter les faits inventés. | `prompts/batch_hallucination_*.txt` |
| `structure` | Prompts pour vérifier la règle "2 Positifs / 1 Négatif". | `prompts/batch_structure_*.txt` |
| `spoiler` | Prompts pour détecter les spoilers narratifs. | `prompts/batch_spoiler_*.txt` |
| `sentiment` | Prompts pour vérifier l'alignement note/texte. | `prompts/batch_sentiment_naive.txt` |
| `sbert_prep` | Prépare l'échantillon pour l'évaluation SBERT (voir ci-dessous). | `prompts/prompt_judge_sbert_300.txt`<br>`csv/sbert_subset_300_stratified.csv` |
| `sbert_eval` | Compare les prédictions SBERT vs Juge (nécessite étape manuelle). | `csv/sbert_evaluation_results_300.csv` |

### 🧠 Workflow Spécifique : Évaluation SBERT
1.  Lancer `python src/evaluation/08_evaluate_quality.py sbert_prep`.
2.  Copier le contenu de `evaluation/prompts/prompt_judge_sbert_300.txt` dans ChatGPT/Claude.
3.  Récupérer **uniquement** le JSON de réponse et le sauvegarder sous `evaluation/csv/judge_labels_300.json`.
4.  Lancer `python src/evaluation/08_evaluate_quality.py sbert_eval`.

---

## 2. Évaluation de la Diversité

**Script :** `09_evaluate_diversity.py`

Mesure si le modèle "tourne en rond". Il analyse la richesse du vocabulaire (n-grams) et la redondance sémantique entre les reviews générées.

**Syntaxe :**
```bash
python src/evaluation/09_evaluate_diversity.py --input [FICHIER_CSV] --save [OPTIONS]
```

### Arguments Clés
*   `--inter-sim` : Active l'analyse sémantique SBERT (recommandé pour détecter si les reviews disent toutes la même chose).
*   `--save` : Ajoute les résultats au rapport `results/results_diversity.md`.
*   `--prefix "Titre"` : Nom de l'expérience dans le rapport.

**Exemple complet :**
```bash
python src/evaluation/09_evaluate_diversity.py \
  --input reports/genai_inputs/prompt_batch_filled.csv \
  --inter-sim \
  --save \
  --prefix "Comparaison Naive vs Engineered"
```

---

## 3. Évaluation du Réalisme & Plagiat

**Script :** `10_evaluate_realism.py`

Compare vos reviews générées avec une base de "vraies" reviews Steam pour voir si elles "sonnent vrai" (proximité sémantique) et vérifier qu'elles ne sont pas du pur copier-coller (plagiat).

**Syntaxe :**
```bash
python src/evaluation/10_evaluate_realism.py --gen [CSV_IA] --real [CSV_REEL] --save [OPTIONS]
```

### Arguments Clés
*   `--gen` : Fichier des reviews générées.
*   `--real` : Fichier des reviews réelles (Ground Truth).
*   `--max-real 2000` : Limite le nombre de reviews réelles utilisées (conseillé pour accélérer le calcul).
*   `--save` : Ajoute les résultats au rapport `results/results_realism.md`.

**Exemple complet :**
```bash
python src/evaluation/10_evaluate_realism.py \
  --gen reports/genai_inputs/prompt_batch_filled.csv \
  --real data/raw/reviews_raw_train.csv \
  --max-real 2000 \
  --save \
  --prefix "Test de Réalisme V1"
```

---

## 📂 Architecture des Sorties

L'exécution de ces scripts peuple automatiquement l'arborescence suivante :

```text
PROJET_RACINE/
├── evaluation/                 # Sorties du script Quality
│   ├── csv/
│   │   ├── sbert_subset_300_stratified.csv
│   │   ├── judge_labels_300.json (Fichier MANUEL)
│   │   └── sbert_evaluation_results_300.csv
│   └── prompts/
│       ├── batch_hallucination_*.txt
│       ├── prompt_judge_sbert_300.txt
│       └── ...
│
└── reports/                    # Entrées et Rapports Markdown
    ├── genai_inputs/           # Fichiers CSV d'entrée
    │   └── prompt_batch_filled.csv
    └── results/                # Rapports cumulatifs (Diversity & Realism)
        ├── results_diversity.md
        └── results_realism.md