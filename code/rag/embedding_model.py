from typing import Sequence
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingModel:
    """
    Local embedding model for RAG-based SQL repair.

    Default usage:
        - all-MiniLM-L6-v2 (fast, 384 dimensions)
        - all-mpnet-base-v2 (slower, higher quality, 768 dimensions)
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> Sequence[float]:
        key = text.strip()
        if key in self.cache:
            return self.cache[key]

        vec = self.model.encode(
            key,
            normalize_embeddings=True
        ).tolist()

        self.cache[key] = vec
        return vec
