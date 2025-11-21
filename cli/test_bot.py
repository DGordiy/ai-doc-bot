# CLI tests will go here
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from backend.services.ai_clients.dummy_local_client import DummyLocalAiClient
from backend.services.document_processor import DocumentProcessor
from backend.services.utils import split_text
from backend.services.vector_store import VectorStore

# Load variables from .env
load_dotenv()

dp = DocumentProcessor()
vs = VectorStore(dim=int(os.getenv("VECTOR_DIM", 768)))
ai = DummyLocalAiClient(api_key=os.getenv("HF_TOKEN"))

# Load documents from tests folder
docs_folder = Path("../tests")
all_texts = []
metadata_list = []
for file_path in docs_folder.iterdir():
    text = dp.process_file(file_path.__str__())
    chunks = split_text(text)

    embeddings = []
    metadata_list = []
    for i, chunk in enumerate(chunks):
        emb = ai.get_embeddings(chunk)
        emb = np.array(emb, dtype=np.float32).reshape(-1)  # ensure 1D
        embeddings.append(emb)
        metadata_list.append({
            "file": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })

    vs.add_texts(chunks, embeddings, metadata_list)
    all_texts.extend(chunks)

print(f"Indexed {len(all_texts)} chunks from {len(list(docs_folder.iterdir()))} files.")

# Search
query = "Is there some meaning?"
query_vector = ai.get_embeddings(query)

# Interactive query loop
print("\n=== AI Document Search ===")
print("Type 'exit' to quit.\n")

while True:
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break
    query_vector = ai.get_embeddings(query)
    results = vs.search(query_vector, top_k=3)
    print("\nTop 3 results:")
    for i, r in enumerate(results, 1):
        meta = r['metadata']
        text = r['text'].replace("\n", " ").strip()
        snippet = text[:200] + ("..." if len(text) > 200 else "")

        print(f"{i}. File: {meta['file']} (chunk {meta['chunk_index'] + 1}/{meta['total_chunks']})")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Snippet: {snippet}\n")