import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_clients.hugging_face_embedding_client import HuggingFaceEmbeddingClient
from backend.services.llm_clients.groq_llm_client import GroqLLMClient
from backend.services.rag.rag_service import RAGService
from backend.services.utils import split_text, truncate_text
from backend.services.vector_store import VectorStore

# ----------------------
# FastAPI App Init
# ----------------------

app = FastAPI(
    title="AI Document Bot",
    description="RAG-based API for document Q&A",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
DOCS_FOLDER = (BASE_DIR / "../../data/samples").resolve()
UPLOAD_DIR = (BASE_DIR / "../../data/uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_DIM = int(os.getenv("VECTOR_DIM", 768))

dp = DocumentProcessor()
emb_client = HuggingFaceEmbeddingClient()
llm_client = GroqLLMClient()

# -------------------------------------
# Initialize Vector Store inside app.state
# -------------------------------------

app.state.vector_store = VectorStore(dim=VECTOR_DIM)


# -------------------------------------
# Helper: index a document
# -------------------------------------

def index_file(file_path: Path):
    text = dp.process_file(str(file_path))
    chunks = split_text(text)
    embeddings = [emb_client.get_embeddings(chunk) for chunk in chunks]

    metadata = [
        {
            "file": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]

    app.state.vector_store.add_texts(chunks, embeddings, metadata)


# -------------------------------------
# Startup: preload test documents
# -------------------------------------

if not DOCS_FOLDER.exists():
    raise RuntimeError(f"Document folder not found: {DOCS_FOLDER}")

for file_path in DOCS_FOLDER.iterdir():
    index_file(file_path)

rag_service = RAGService(emb_client, llm_client, app.state.vector_store)


# ----------------------
# Request Models
# ----------------------

class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is the purpose of this document?")
    top_k: int = Field(5, example=5)


class SearchResponseItem(BaseModel):
    text: str
    file: str
    chunk: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResponseItem]


class AnswerResponse(BaseModel):
    answer: str


# ----------------------
# API Endpoints
# ----------------------

@app.post("/search", response_model=SearchResponse)
async def search_docs(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Query cannot be empty")

    query_vector = emb_client.get_embeddings(req.question)
    results = app.state.vector_store.search(query_vector, top_k=req.top_k)

    formatted = [
        SearchResponseItem(
            text=r["text"],
            file=r["metadata"]["file"],
            chunk=f"{r['metadata']['chunk_index'] + 1}/{r['metadata']['total_chunks']}",
            score=r["score"]
        )
        for r in results
    ]

    return SearchResponse(results=formatted)


@app.post("/rag_answer", response_model=AnswerResponse)
async def rag_answer(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    answer = rag_service.rag_answer(req.question, top_k=req.top_k)
    return AnswerResponse(answer=answer)


@app.get("/list_documents")
async def list_documents():
    files = sorted({m["file"] for m in app.state.vector_store.metadata})
    return {"documents": files}


@app.get("/debug/chunks")
async def debug_chunks(file: str):
    chunks = []
    vs = app.state.vector_store
    for text, meta in zip(vs.texts, vs.metadata):
        if meta["file"] == file:
            chunks.append({
                "chunk_index": meta["chunk_index"],
                "total_chunks": meta["total_chunks"],
                "text": truncate_text(text, 300)
            })

    if not chunks:
        raise HTTPException(404, f"No chunks found for file: {file}")

    return {"file": file, "chunks": chunks}


@app.post("/upload_file")
async def upload_file(file: UploadFile):
    allowed = {".pdf", ".txt", ".docx"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    saved_path = UPLOAD_DIR / file.filename
    content = await file.read()

    with open(saved_path, "wb") as f:
        f.write(content)

    index_file(saved_path)

    return {
        "status": "ok",
        "file": file.filename,
        "indexed": True
    }


@app.delete("/delete_file")
async def delete_file(filename: str):
    # full path to file
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, f"File '{filename}' not found in uploads")

    # remove file
    file_path.unlink()

    # rebuild index
    app.state.vector_store = VectorStore(dim=VECTOR_DIM)

    # reindex test docs + uploaded docs
    for folder in [DOCS_FOLDER, UPLOAD_DIR]:
        for f in folder.iterdir():
            index_file(f)

    return {"status": "deleted", "file": filename}


@app.post("/rebuild_index")
async def rebuild_index():
    app.state.vector_store = VectorStore(dim=VECTOR_DIM)

    # reindex everything
    for folder in [DOCS_FOLDER, UPLOAD_DIR]:
        for file_path in folder.iterdir():
            index_file(file_path)

    return {
        "status": "index rebuilt",
        "documents_count": len(list(DOCS_FOLDER.iterdir())) + len(list(UPLOAD_DIR.iterdir()))
    }
