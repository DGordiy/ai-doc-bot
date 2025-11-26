import logging
import os

from dotenv import load_dotenv
from groq import Groq

from backend.services.llm_clients.base_llm_client import BaseLLMClient

logger = logging.Logger(__name__)

load_dotenv()


class GroqLLMClient(BaseLLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        super().__init__(model)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )

        return chat_completion.choices[0].message.content
