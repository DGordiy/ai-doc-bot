from backend.services.llm_clients.hugging_face_llm_client import HuggingFaceLLMClient


def main():
    client = HuggingFaceLLMClient()
    prompt = "Explain in simple words what embeddings are."
    answer = client.generate(prompt)
    print("=== Generated Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
