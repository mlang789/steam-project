# Documentation Complète du Projet : Steam Reviews GenAI

Ce projet est un pipeline complet de Machine Learning et d'IA Générative.
Il est structuré en 3 grandes phases :
1.  **Data Engineering** : Collecte et préparation des données Steam.
2.  **Juge IA** : Entraînement de classifieurs (SBERT & TF-IDF) pour évaluer la qualité.
3.  **GenAI** : Fine-tuning d'un LLM et génération de reviews.

---

## Structure du Projet

Voici l'organisation des fichiers :

```text
steam-project/
├── data/                      # Stockage des données
│   ├── raw/                   # JSON/CSV bruts de l'API Steam
│   ├── processed/             # Données nettoyées (dataset_train.csv, titles.csv)
│   └── outputs_gen_ai_models/ # Réception des fichiers venant de Colab
├── reports/                   # Artefacts et Résultats
│   ├── models/                # Modèles du Juge (.joblib)
│   ├── genai_inputs/          # Fichiers d'entrée pour le LLM (prompts, jsonl)
│   └── evaluation/            # Rapports de performance
├── src/                       # Code Source
│   ├── config.py              # Configuration globale
│   ├── utils.py               # Utilitaires (SBERT, etc.)
│   ├── data/                  # Pipeline ETL (Extract-Transform-Load)
│   ├── judge/                 # Classification (Le Juge)
│   └── genai/                 # Pipeline genai (Entrainement & Inférence)
```

---

## Phase 1 : Pipeline de Données

Cette phase transforme les données brutes d'Internet en un dataset propre pour le Machine Learning.

### Étape 1 : Collecte (`01_collect.py`)

Télécharge les reviews et les titres des jeux depuis l'API Steam.

*   **Commandes :**
    ```bash
    python src/data/01_collect.py train
    python src/data/01_collect.py validation
    ```
*   **Outputs :**
    *   `data/raw/reviews_raw_train.csv`
    *   `data/processed/titles_train.csv` (Titres des jeux)

### Étape 2 : Construction du Dataset (`02_process.py`)

Nettoie les textes, fusionne avec les titres, et convertit le vote (recommandé/non) en note (3/10 ou 9/10).

*   **Commandes :**
    ```bash
    python src/data/02_process.py train
    python src/data/02_process.py validation
    ```
*   **Outputs :**
    *   `data/processed/dataset_train.csv` : **Le fichier maître**.
    *   `data/processed/dataset_val.csv` : Pour tests finaux.

---

## Phase 2 : Entraînement du Juge   

Création d'une IA capable de dire si une review est positive ou négative. Nous entraînons deux versions pour comparer l'évaluation.

### Étape 3 : Entraînement (`03_train.py`)

#### Option A : Modèle SBERT (Recommandé)
Capture la sémantique (le sens) des phrases.

*   **Commande :**
    ```bash
    python src/judge/03_train.py --model sbert
    ```
*   **Output :** `reports/models/judge_model_sbert.joblib`

#### Option B : Modèle TF-IDF (Baseline)
Se base sur les mots-clés. Utile pour vérifier si le LLM "triche" en utilisant juste des mots positifs.

*   **Commande :**
    ```bash
    python src/judge/03_train.py --model tfidf
    ```
*   **Output :** `reports/models/judge_model_tfidf.joblib`

---

## Phase 3 : Generative AI (Local & Colab)

Cette phase est hybride : préparation en local, entraînement/génération sur Google Colab (GPU), puis retour en local pour l'évaluation.

### Étape 4 : Préparation des Données (Local)

#### A. Création des Prompts "Zero-Shot" (`04_generate_prompts.py`)
Génère les instructions que le LLM devra suivre (Naive vs Engineered).

*   **Commande :**
    ```bash
    python src/genai/04_generate_prompts.py
    ```
*   **Output :** `reports/genai_inputs/prompt_batch.csv`

#### B. Préparation du Dataset de Fine-Tuning (`05_prepare_training.py`)
Transforme les vraies reviews en format JSONL pour ré-entraîner le LLM (Instruction Tuning).

*   **Commande :**
    ```bash
    python src/genai/05_prepare_training.py
    ```
*   **Output :** `reports/genai_inputs/sft_train.jsonl`

---

### Étape 5 : Exécution sur Google Colab (GPU requis) 

Utilisez les fichiers générés à l'étape 4.

#### Notebook 1 : Entraînement (Fine-Tuning LoRA)
1.  **Upload :** Déposez `reports/genai_inputs/sft_train.jsonl` sur le Colab.
2.  **Entraînement :** Lancez le notebook pour fine-tuner `TinyLlama-1.1B`.
3.  **Sauvegarde :** Récupérez le dossier de l'adaptateur LoRA sur votre Google Drive.

#### Notebook 2 : Inférence & Génération
1.  **Upload :** Déposez `reports/genai_inputs/prompt_batch.csv` sur le Colab.
2.  **Génération :** Lancez le notebook pour générer les 3 fichiers CSV :
    *   `generated_reviews_naive.csv`
    *   `generated_reviews_engineered.csv`
    *   `generated_reviews_finetuned.csv`
3.  **Download :** Téléchargez ces 3 fichiers sur votre ordinateur local.

---

### Étape 6 : Consolidation et Évaluation (Local)

#### C. Fusion des Résultats (`06_merge_outputs.py`)
1.  **Action :** Placez les 3 fichiers CSV du Colab dans `data/outputs_genai_models/`.
2.  **Commande :**
    ```bash
    python src/genai/06_merge_outputs.py
    ```
3.  **Output :** `reports/genai_inputs/prompt_batch_filled.csv`

#### D. Évaluation Finale (`07_evaluate.py`)
Le "Juge" note automatiquement vos reviews générées. Vous pouvez choisir quel juge utiliser.

*   **Avec le juge SBERT (Sémantique) :**
    ```bash
    python src/genai/07_evaluate.py --prefix eval_finale --model sbert
    ```
    *Output : `reports/evaluation/eval_finale_sbert_summary.csv`*

*   **Avec le juge TF-IDF (Mots-clés) :**
    ```bash
    python src/genai/evaluate_generations.py --prefix eval_finale --model tfidf
    ```
    *Output : `reports/evaluation/eval_finale_tfidf_summary.csv`*

---

## Résumé du Flux de Données

```mermaid
graph TD
    A[API Steam] -->|src/data_pipeline| B(data/processed/dataset_train.csv)
    
    B -->|src/judge| C[reports/models/judge_model_sbert.joblib]
    B -->|src/judge| D[reports/models/judge_model_tfidf.joblib]
    
    B -->|src/genai| E[reports/genai_inputs/sft_train.jsonl]
    B -->|src/genai| F[reports/genai_inputs/prompt_batch.csv]
    
    E & F -.->|Google Colab| G[data/outputs_genai_models/*.csv]
    
    G -->|src/genai/merge| H(prompt_batch_filled.csv)
    
    H & C -->|src/genai/eval| I[Rapport SBERT]
    H & D -->|src/genai/eval| J[Rapport TF-IDF]
```