import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        # FAISS index for vectors of necessary dimension
        self.index = faiss.IndexFlatL2(dim) # L2 - Euclidean distance
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add_texts(self, texts: list[str], embeddings: list[list[float]], metadata: list[dict]):
        """Adding texts and vectors to index
        :param metadata:
        :param texts:
        :param embeddings:
        """
        vectors = np.array(embeddings, dtype=np.float32)
        assert len(texts) == len(metadata)
        assert vectors.shape[1] == self.index.d

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_vector: list[float], top_k: int = 5):
        """Search for top-k nearest texts
        :param query_vector:
        :param top_k:
        """
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(dist)
            })
        return results