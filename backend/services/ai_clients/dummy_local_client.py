import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.ai_clients.ai_client import AIClient


class DummyLocalAiClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, text: str) -> list[float]:
        emb = self.model.encode(text, convert_to_numpy=True)  # shape (384,)
        return emb.astype(np.float32).tolist()
