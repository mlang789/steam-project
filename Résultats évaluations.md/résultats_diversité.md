=== Diversity evaluation summary (by method) ===
               method    n  mean_word_count  distinct1_mean  distinct2_mean  trigram_repeat_mean  inter_sim_mean
base_engineered_clean 1800       121.461667        0.802228        0.990125             0.000479        0.836884

=== Diversity evaluation summary (by method) ===
      method    n  mean_word_count  distinct1_mean  distinct2_mean  trigram_repeat_mean  inter_sim_mean
finetuned_v5 4416       116.036685         0.77216        0.980973             0.001183        0.773702

=== Diversity evaluation summary (by method) ===
          method    n  mean_word_count  distinct1_mean  distinct2_mean  trigram_repeat_mean  inter_sim_mean
base_naive_clean 1800       121.812778        0.801443        0.991491             0.000202        0.841343

Les résultats montrent que le fine-tuning entraîne une légère diminution de la diversité lexicale locale, mais améliore la diversité globale entre reviews, tandis que les approches par prompt génèrent des textes plus variés localement mais plus homogènes dans leur contenu global.

