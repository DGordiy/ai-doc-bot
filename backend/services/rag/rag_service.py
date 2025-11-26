from backend.services.embedding_clients.base_embedding_client import BaseEmbeddingClient
from backend.services.llm_clients.base_llm_client import BaseLLMClient
from backend.services.rag.rag_builder import build_rag_context
from backend.services.vector_store import VectorStore


class RAGService:
    def __init__(self, embedding_client: BaseEmbeddingClient, llm_client: BaseLLMClient, vector_store: VectorStore):
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.vector_store = vector_store

    def rag_answer(self, question: str, top_k: int = 5) -> str:
        # Get embeddings from the question
        query_vector = self.embedding_client.get_embeddings(question)

        # Retrieve similar docs
        results = self.vector_store.search(query_vector, top_k=top_k)

        # Build formatted RAG context
        context_str = build_rag_context(results)

        system_prompt = ("You are a helpful assistant. "
                         "You MUST answer strictly and ONLY using the information contained in the provided document excerpts. "
                         "If the required information is not present in the excerpts, respond exactly with: "
                         "'The document does not contain this information.' "
                         "Do not use external knowledge. Do not guess. "
                         "Keep the answer factual, concise, and directly based on the excerpts.")

        user_prompt = (f"Context:\n\n{context_str}\n\n"
                       f"Question: {question}\n\n"
                       f"Give a short factual answer based ONLY on the context.")

        return self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
