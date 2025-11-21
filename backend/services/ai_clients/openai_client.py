from openai import OpenAI

from backend.services.ai_clients.ai_client import AIClient


class OpenAiClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"


    def get_embeddings(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input = [text], model=self.model)
        return response.data[0].embedding