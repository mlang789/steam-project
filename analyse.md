# Analyse des Performances des Modèles GenAI

Ce document présente une évaluation approfondie des trois approches de génération de reviews Steam (Naïve, Engineered, Fine-tuned). L'analyse porte sur la qualité factuelle, le respect des structures logiques, la cohérence sémantique, ainsi que le réalisme et la diversité des textes produits.

## 1. Analyse des Hallucinations et de la Qualité Factuelle

Cette section mesure la tendance des modèles à inventer des faits inexistants (mécaniques de jeu, modes, personnages).

### Comparaison des taux d'hallucination

| Modèle | Jeu | Taux d'Hallucination | Moyenne Globale | Principales Typologies d'Erreurs |
| :--- | :--- | :---: | :---: | :--- |
| **Naïf** (Zéro-Shot) | TF2 <br> CS2 | 56% <br> 60% | **58%** | **Confusion de genre :** invention de modes "histoire", mécaniques RPG (niveaux, arbres de compétences) et personnages de jeux tiers. |
| **Engineered** (Instructions) | TF2 <br> CS2 | 68% <br> 62% | **65%** | **Détails fictifs accrus :** invention de classes, de développeurs et de mécaniques complexes (grappins, selfies). |
| **Fine-tuned** (Entraîné) | TF2 <br> CS2 | 28% <br> 34% | **31%** | **Résidus techniques :** fuites de métadonnées, confusions de plateformes ou chiffres erronés, mais respect global du genre. |

**Observations majeures :**

1.  **Le paradoxe de l'ingénierie (+7%) :** Le modèle "Engineered" présente le taux d'erreur le plus élevé (65%). En forçant le modèle à être spécifique via le prompt, on le contraint à générer des détails qu'il ne maîtrise pas, provoquant une sur-créativité factuellement fausse.
2.  **L'efficacité du fine-tuning (-34%) :** L'entraînement sur de réelles données réduit le taux d'hallucination de plus de moitié. Le modèle intègre le lexique et les limites du jeu, éliminant les inventions hors-sujet comme les mondes ouverts dans des jeux de tir.
3.  **Thématiques récurrentes :** Les modèles non entraînés tendent à transformer tout jeu en RPG narratif. Le modèle fine-tuned corrige ce biais mais souffre parfois de pollution de données (apparitions d'identifiants ou d'artefacts techniques).

---

## 2. Respect des Instructions Logiques et Structurelles

Cette partie évalue la capacité des modèles à suivre des contraintes strictes : structure argumentative (2 points positifs, 1 négatif) et interdiction de spoilers.

### Comparaison de l'adhérence au prompt

| Modèle | Jeu | Échec de Structure (2+ / 1-) | Taux de "Faux Spoilers" | Observation Clé |
| :--- | :--- | :---: | :---: | :--- |
| **Naïf** | Dota 2 <br> CS2 | N/A (Prompt libre) | 14% <br> 22% | Invente des scénarios par réflexe narratif. |
| **Engineered** | Dota 2 <br> CS2 | **100%** <br> **96%** | 10% <br> 36% | **Échec quasi-total** sur la logique quantitative. |
| **Fine-tuned** | Dota 2 <br> CS2 | **90%** <br> **82%** | 4% <br> 10% | **Amélioration nette** de la compréhension structurelle. |

**Analyse :**

*   **Limites du prompt seul :** L'instruction explicite "exactement 2 points positifs et 1 négatif" échoue dans 98% des cas avec le modèle Engineered. Les modèles de taille réduite (1.1B paramètres) privilégient la fluidité linguistique à la rigueur arithmétique.
*   **Apport du fine-tuning :** Le taux de réussite sur la structure passe de ~2% à ~14%. Bien que le score reste modeste, le modèle tente d'équilibrer ses arguments.
*   **Gestion des spoilers :** L'instruction "pas de spoilers" a paradoxalement incité les modèles Naïf et Engineered à inventer une histoire pour pouvoir en parler. Le fine-tuning corrige ce défaut : ayant appris que des jeux comme CS2 n'ont pas de campagne narrative, le taux d'hallucination narrative chute drastiquement (de 36% à 4% sur CS2).

---

## 3. Cohérence Sémantique (Note vs Texte)

Nous analysons ici l'alignement entre la note attribuée (input) et le sentiment dégagé par le texte généré.

### Taux de conformité entre note cible et sentiment

| Jeu Évalué | Note Cible | Sentiment Attendu | Taux de Succès | Nature des Divergences |
| :--- | :---: | :---: | :---: | :--- |
| **Dota 2** | 9 / 10 | POSITIF | **100%** | Aucune. |
| **Dota 2** | 3 / 10 | NÉGATIF | **80%** | Incohérence interne (ex: "étonné par la qualité" pour un 3/10). |
| **Team Fortress 2** | 9 / 10 | POSITIF | **100%** | Aucune. |
| **Team Fortress 2** | 3 / 10 | NÉGATIF | **84%** | Conflit affectif (ex: "mon jeu préféré" pour un 3/10). |
| **Counter-Strike 2** | 3 / 10 | NÉGATIF | **76%** | Hallucinations de satisfaction. |
| **MOYENNE** | - | - | **88.0%** | **Biais de positivité prédominant.** |

**Interprétation :**
Un biais de complaisance est observé. Le modèle excelle dans la génération positive (100% de succès) mais montre une résistance à la critique. Environ 20% des reviews négatives contiennent des éloges contradictoires. Cela suggère que l'alignement initial du modèle de base (TinyLlama) favorise un ton utile et enthousiaste, créant une dissonance cognitive lors de la génération de critiques sévères.

---

## 4. Analyse du Réalisme et de la Diversité

Cette section compare la proximité des textes avec de vraies reviews humaines et leur variété lexicale.

### Synthèse du réalisme (Sémantique)
*   **Naïf & Engineered (Score ~0.69) :** Ces approches produisent des textes originaux qui ne copient pas le corpus existant. Aucun cas de plagiat n'est détecté.
*   **Fine-tuned (Score ~0.70) :** Le fine-tuning offre le réalisme sémantique le plus élevé. Une infime trace de mémorisation (0.02% de chevauchement) est détectée, ce qui est négligeable mais indique que le modèle commence à retenir des phrases types du dataset d'entraînement.

### Synthèse de la diversité
*   **Diversité Lexicale (Interne) :** Les modèles Naïf et Engineered utilisent un vocabulaire plus varié localement (plus de mots uniques par review). Le fine-tuning réduit légèrement cette variété lexicale.
*   **Diversité Sémantique (Globale) :**
    *   Les modèles à prompts (Naïf/Engineered) ont une similarité inter-review élevée (~0.84). Ils tendent à répéter les mêmes structures et arguments pour tous les jeux.
    *   Le modèle **Fine-tuned** présente une similarité plus faible (~0.77). Cela signifie qu'il génère des reviews plus distinctes les unes des autres.

**Conclusion sur la diversité :** Le fine-tuning produit des textes individuellement plus simples, mais globalement plus variés et moins répétitifs que les approches basées sur le prompt engineering.

---

## 5. Fiabilité de l'Évaluation Automatisée (Juge SBERT)

Pour valider nos métriques, nous avons comparé les prédictions du classifieur interne (SBERT) avec celles d'un juge expert (LLM externe) sur un échantillon de 300 reviews.

| Segment de Données | Accuracy | F1-Score (Positif) | F1-Score (Négatif) | Fiabilité |
| :--- | :---: | :---: | :---: | :--- |
| **Global** | **87.67%** | **0.92** | **0.69** | **Élevée** |
| **Sous-groupe : Naïf** | 93.00% | 0.96 | 0.81 | Très Élevée |
| **Sous-groupe : Engineered** | 84.00% | 0.89 | 0.62 | Modérée |
| **Sous-groupe : Fine-tuned** | 86.00% | 0.91 | 0.65 | Élevée |

**Validité de l'outil :**
Le juge SBERT est robuste pour l'évaluation globale. On note cependant un déséquilibre : il détecte mieux les avis positifs (F1: 0.92) que les avis négatifs (F1: 0.69), souvent plus nuancés ou sarcastiques. La précision baisse légèrement sur les textes complexes (Engineered/Fine-tuned) par rapport aux textes stéréotypés du modèle Naïf.