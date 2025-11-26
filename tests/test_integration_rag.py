import os
from pathlib import Path

from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_clients.hugging_face_embedding_client import HuggingFaceEmbeddingClient
from backend.services.llm_clients.base_llm_client import BaseLLMClient
from backend.services.rag.rag_service import RAGService
from backend.services.utils import split_text
from backend.services.vector_store import VectorStore


class DummyLLMClient(BaseLLMClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "This is a dummy answer based on context."


def test_full_rag_workflow():
    # Prepare components
    dp = DocumentProcessor()
    vs = VectorStore(dim=int(os.getenv("VECTOR_DIM", 768)))
    emb_client = HuggingFaceEmbeddingClient()
    llm_client = DummyLLMClient(model="dummy")
    rag = RAGService(embedding_client=emb_client, llm_client=llm_client, vector_store=vs)

    # Load sample documents
    docs_folder = Path("tests/samples")
    for file_path in docs_folder.iterdir():
        text = dp.process_file(str(file_path))
        chunks = split_text(text)
        embeddings = [emb_client.get_embeddings(chunk) for chunk in chunks]
        metadata = [{"file": file_path.name, "chunk_index": i, "total_chunks": len(chunks)} for i, chunk in
                    enumerate(chunks)]
        vs.add_texts(chunks, embeddings, metadata)

    # Run RAG
    question = "What is the main idea of the documents?"
    answer = rag.rag_answer(question)

    assert answer == llm_client.generate('', '')
