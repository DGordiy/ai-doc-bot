# -----------------------
# Tests for rag_builder.py
# -----------------------
from unittest.mock import MagicMock

from backend.services.rag.rag_builder import truncate_text, build_rag_context
from backend.services.rag.rag_service import RAGService


def test_truncate_text_short():
    text = "Hello"
    assert truncate_text(text, max_chars=10) == text


def test_truncate_text_long():
    text = "a" * 20
    truncated = truncate_text(text, max_chars=10)
    assert truncated == "a" * 10 + "..."
    assert len(truncated) == 13


def test_build_rag_context_respects_max_total_chars():
    results = [
        {"text": "a" * 500, "metadata": {"file": "doc1.txt", "chunk_index": 0, "total_chunks": 2}},
        {"text": "b" * 500, "metadata": {"file": "doc2.txt", "chunk_index": 1, "total_chunks": 2}},
    ]

    context = build_rag_context(results, max_total_chars=600)

    # Should include only first block
    assert "File: doc1.txt" in context
    assert "File: doc2.txt" not in context
    assert len(context) <= 600 + len("\n\n---\n\n")  # учёт разделителя


# -----------------------
# Tests for RAGService
# -----------------------

def test_rag_answer_calls_llm():
    # Моск embedding_client
    embedding_client = MagicMock()
    embedding_client.get_embeddings.return_value = [0.1, 0.2, 0.3]

    # Моск vector_store
    vector_store = MagicMock()
    vector_store.search.return_value = [
        {"text": "Some text from document", "metadata": {"file": "file.txt", "chunk_index": 0, "total_chunks": 1},
         "score": 0.01}
    ]

    # Моск llm_client
    llm_client = MagicMock()
    llm_client.generate.return_value = "LLM answer based on context"

    rag = RAGService(embedding_client, llm_client, vector_store)

    question = "What is in the document?"
    answer = rag.rag_answer(question)

    # Check that LLM called and answered correctly
    llm_client.generate.assert_called_once()
    assert answer == "LLM answer based on context"

    # Check that embedding_client called with question
    embedding_client.get_embeddings.assert_called_once_with(question)

    # Check that vector_store.search called with vector
    vector_store.search.assert_called_once()
