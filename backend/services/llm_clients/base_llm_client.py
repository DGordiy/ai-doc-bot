from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    def __init__(self, model: str):
        self._client = None
        self.model = model

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, client):
        self._client = client

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass
