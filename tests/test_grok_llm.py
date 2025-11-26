from backend.services.llm_clients.groq_llm_client import GroqLLMClient


def main():
    client = GroqLLMClient()
    prompt = "Explain in simple words what embeddings are."
    answer = client.generate(prompt)
    print("=== Generated Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
