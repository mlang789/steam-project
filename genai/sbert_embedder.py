import numpy as np
from sentence_transformers import SentenceTransformer

class SBERTEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        emb = self.model.encode(
            list(X),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb)
