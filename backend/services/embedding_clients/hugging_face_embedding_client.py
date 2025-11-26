import os

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.embedding_clients.base_embedding_client import BaseEmbeddingClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class HuggingFaceEmbeddingClient(BaseEmbeddingClient):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, text: str) -> list[float]:
        emb = self.model.encode(text, convert_to_numpy=True)  # shape (384,)
        return emb.astype(np.float32).tolist()
