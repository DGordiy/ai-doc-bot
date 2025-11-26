from abc import ABC, abstractmethod


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        pass