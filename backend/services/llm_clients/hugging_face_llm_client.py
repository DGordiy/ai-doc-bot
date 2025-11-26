import logging
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from backend.services.llm_clients.base_llm_client import BaseLLMClient

logger = logging.Logger(__name__)

load_dotenv()


class HuggingFaceLLMClient(BaseLLMClient):
    def __init__(self, model: str = "sst5-tiny"): # here should be model which provides text generation
        super().__init__(model)
        self.client = InferenceClient(model=model, token=os.getenv("HF_API_KEY"))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.text_generation(system_prompt, user_prompt)
        return response
