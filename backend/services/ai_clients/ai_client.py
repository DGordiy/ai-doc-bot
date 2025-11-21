from abc import ABC, abstractmethod


class AIClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        pass