# CLI tests will go here
import os
from pathlib import Path

from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_clients.hugging_face_embedding_client import HuggingFaceEmbeddingClient
from backend.services.llm_clients.groq_llm_client import GroqLLMClient
from backend.services.rag.rag_service import RAGService
from backend.services.utils import split_text
from backend.services.vector_store import VectorStore


def main():
    print("=== Building test RAG index ===")

    # -------- 1. Prepare components --------
    dp = DocumentProcessor()
    vs = VectorStore(dim=int(os.getenv("VECTOR_DIM", 768)))
    emb_client = HuggingFaceEmbeddingClient()
    llm_client = GroqLLMClient()
    rag = RAGService(embedding_client=emb_client, llm_client=llm_client, vector_store=vs)

    # Load documents from tests folder
    docs_folder = Path("../tests/samples")
    all_texts = []
    metadata_list = []
    for file_path in docs_folder.iterdir():
        print(f"Processing file: {file_path.name}")

        text = dp.process_file(str(file_path))
        chunks = split_text(text)

        embeddings = []
        metadata_list = []
        for i, chunk in enumerate(chunks):
            emb = emb_client.get_embeddings(chunk)
            embeddings.append(emb)
            metadata_list.append({
                "file": file_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        vs.add_texts(chunks, embeddings, metadata_list)
        all_texts.extend(chunks)

    print("Indexing complete.\n")

    # -------- 3. Ask questions interactively --------
    print("=== RAG Chat ===")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ")
        if question.lower().strip() == "exit":
            break

        answer = rag.rag_answer(question)
        print("\nAnswer:")
        print(answer)
        print("\n---------------------\n")


if __name__ == "__main__":
    main()
