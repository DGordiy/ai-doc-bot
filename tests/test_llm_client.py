from backend.services.llm_clients.base_llm_client import BaseLLMClient


class DummyLLMClient(BaseLLMClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Dummy answer"


def test_dummy_llm_client_generate():
    client = DummyLLMClient(model="dummy-model")
    result = client.generate(system_prompt="system", user_prompt="user")
    assert result == "Dummy answer"
