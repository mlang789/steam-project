# Analyse de sentiment et génération conditionnelle de reviews steam

Ce projet explore les capacités des techniques modernes de traitement du langage naturel (nlp) appliquées aux critiques de jeux vidéo sur la plateforme steam. Il s'articule autour de deux axes de recherche complémentaires : la compréhension du texte via la classification et la production de contenu synthétique via des modèles génératifs.

L'objectif principal est de déterminer dans quelle mesure une intelligence artificielle peut imiter le style, le vocabulaire et les biais spécifiques d'une communauté de joueurs. au-delà de la simple génération de texte, ce projet cherche à évaluer rigoureusement comment différentes stratégies de contrôle (prompting vs fine-tuning) influencent la véracité et la structure des critiques générées.

Le projet suit une méthodologie progressive : nous commençons par établir une baseline de prédiction de sentiment, puis nous construisons et comparons plusieurs systèmes de génération de texte conditionnés par le titre d'un jeu et une note cible.

## Structure du dépôt

L'organisation des fichiers sépare clairement le code source, les données et les analyses.

```text
steam-project/
├── USAGE.md           # rapport complet et détaillé de l'execution des différents fichiers
├── analyse.md         # rapport complet et détaillé des résultats de génération
├── notebooks/         # notebooks jupyter (exploration) et google colab (fine-tuning gpu)
├── data/              # données locales (ignorées par git)
├── reports/           # modèles sauvegardés, prompts et résultats d'évaluation
├── src/               # code source python modulaire
│   ├── data/          # scripts de collecte et nettoyage
│   ├── judge/         # entraînement des classifieurs
│   ├── genai/         # pipeline de génération et préparation du fine-tuning
│   └── evaluation/    # scripts d'analyse quantitative
└── requirements.txt   # dépendances du projet
```

Les analyses approfondies concernant les performances des modèles génératifs, notamment l'étude des hallucinations et des métriques de diversité, sont disponibles dans le fichier `analyse.md` à la racine.

Les notebooks nécessaires pour reproduire les entraînements sur gpu (google colab) ou visualiser les données sont situés dans le dossier `notebooks/`.

## Installation

Le projet nécessite python 3.10 ou une version ultérieure. l'installation des dépendances se fait via la commande suivante :

```bash
pip install -r requirements.txt
```

## Données

Les données utilisées proviennent de l'api publique de steam. chaque critique est composée du texte de l'avis, d'un label de recommandation binaire et de métadonnées contextuelles.

Pour les besoins de la génération conditionnelle, nous transformons la recommandation binaire en une note sur 10 pour nuancer l'instruction donnée au modèle :
*   recommandé : correspond à une note de 9/10 (sentiment très positif).
*   non recommandé : correspond à une note de 3/10 (sentiment négatif mais argumenté).

Les scripts fournis dans `src/data/` permettent de reconstituer intégralement les datasets localement, car les données brutes ne sont pas stockées sur le dépôt.

## Méthodologie et approches

Nous mettons en concurrence trois paradigmes de génération pour produire des critiques artificielles :

1.  **approche naïve :** nous utilisons un grand modèle de langage avec une instruction minimaliste demandant simplement de rédiger un avis pour un jeu donné.
2.  **prompt engineering :** nous complexifions l'instruction en imposant des contraintes structurelles strictes (nombre de mots, structure argumentative précise, interdiction de spoilers) pour tenter de guider le modèle sans le ré-entraîner.
3.  **fine-tuning (lora) :** nous ré-entraînons un modèle plus modeste sur un corpus de critiques réelles. l'objectif est d'apprendre par l'exemple le style "gamer" et les spécificités des jeux, plutôt que de suivre des règles explicites.

## Utilisation du pipeline

L'exécution complète du projet se déroule en quatre étapes successives.

### 1. collecte et préparation
Ces commandes téléchargent les données brutes et créent les fichiers csv nettoyés.

```bash
python src/data/01_collect.py train
python src/data/01_collect.py validation
python src/data/02_process.py train
python src/data/02_process.py validation
```

### 2. entraînement du juge
Nous créons un modèle capable d'évaluer automatiquement si une critique générée est positive ou négative. cela servira de métrique de contrôle.

```bash
python src/judge/03_train.py --model sbert
```

### 3. génération de texte
Cette phase prépare les fichiers d'instructions.
*   génération des prompts pour les approches sans entraînement :
    ```bash
    python src/genai/04_generate_prompts.py
    ```
*   préparation des données pour le fine-tuning :
    ```bash
    python src/genai/05_prepare_training.py
    ```

Une fois ces fichiers générés, les notebooks présents dans `notebooks/` doivent être utilisés (par exemple sur colab) pour effectuer l'inférence et l'entraînement gpu. les résultats doivent ensuite être fusionnés :

```bash
python src/genai/06_merge_outputs.py
```

### 4. Évaluation

Enfin, nous mesurons la qualité des productions. Pour ne pas alourdir ce document principal, la liste exhaustive des commandes d'évaluation est détaillée dans le fichier [USAGE.md](./USAGE.md).

## Synthèse des résultats

L'étude comparative, détaillée dans `analyse.md`, met en évidence des différences marquées entre les approches. le fine-tuning s'avère être la méthode la plus efficace pour réduire les hallucinations factuelles (invention de modes de jeu inexistants). à l'inverse, l'ingénierie de prompt, bien qu'utile pour la mise en forme, tend paradoxalement à augmenter le taux d'erreur factuelle en forçant le modèle à inventer des détails pour satisfaire les contraintes de longueur.

Le code est fourni à titre éducatif pour illustrer un pipeline complet de machine learning, de l'extraction de données à l'évaluation par des juges automatisés.