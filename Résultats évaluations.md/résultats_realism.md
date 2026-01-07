=== Realism evaluation summary (by method) ===
          method  realism_mean  nn_overlap_rate  nn_overlap_max
base_naive_clean      0.690767              0.0         0.89179

Les prompts naïfs produisent des reviews sémantiquement proches des reviews Steam réelles, tout en conservant une forte originalité : aucun cas de similarité excessive n’est détecté, ce qui indique l’absence de mémorisation ou de copiage.

=== Realism evaluation summary (by method) ===
               method  realism_mean  nn_overlap_rate  nn_overlap_max
base_engineered_clean      0.684888              0.0        0.881317 

Les prompts engineered produisent des reviews sémantiquement réalistes, proches du style des reviews Steam réelles, tout en conservant une forte originalité. Aucune génération ne présente de similarité excessive avec le corpus réel, ce qui indique l’absence de mémorisation et un bon contrôle du processus de génération.

=== Realism evaluation summary (by method) ===
      method  realism_mean  nn_overlap_rate  nn_overlap_max
finetuned_v5      0.697258         0.000226        0.902977

Le fine-tuning permet d’augmenter le réalisme sémantique des reviews générées, en les rapprochant fortement des reviews Steam réelles. Cette amélioration s’accompagne toutefois d’un risque très marginal de similarité excessive, visible sur un nombre extrêmement limité de générations, ce qui suggère un léger effet de mémorisation sans compromettre l’originalité globale.

