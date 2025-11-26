import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.services.embedding_clients.base_embedding_client import BaseEmbeddingClient

load_dotenv()

class OpenBaseEmbeddingEmbeddingClient(BaseEmbeddingClient):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
        self.model = "text-embedding-3-small"


    def get_embeddings(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input = [text], model=self.model)
        return response.data[0].embedding