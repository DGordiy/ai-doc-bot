import numpy as np
import pytest

from backend.services.vector_store import VectorStore


@pytest.fixture
def dummy_data():
    texts = ["Hello world", "Python is great", "AI is awesome"]
    embeddings = [np.random.rand(768).tolist() for _ in texts]
    metadata = [{"file": f"file{i}.txt", "chunk_index": i, "total_chunks": len(texts)} for i in range(len(texts))]
    return texts, embeddings, metadata


@pytest.fixture
def vector_store():
    return VectorStore(dim=768)


def test_add_texts(vector_store, dummy_data):
    texts, embeddings, metadata = dummy_data
    vector_store.add_texts(texts, embeddings, metadata)

    assert len(vector_store.texts) == len(texts)
    assert len(vector_store.metadata) == len(metadata)
    assert vector_store.index.ntotal == len(texts)


def test_search(vector_store, dummy_data):
    texts, embeddings, metadata = dummy_data
    vector_store.add_texts(texts, embeddings, metadata)

    # Using embedding from first text for search
    query_vector = embeddings[0]
    results = vector_store.search(query_vector, top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == texts[0]
    assert results[0]["metadata"]["file"] == "file0.txt"
    assert "score" in results[0]
    assert "metadata" in results[0]
