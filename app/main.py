from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .rag import RAGStore

app = FastAPI(title="RAG Document Q&A", version="1.0.0")
_store: RAGStore | None = None


def store() -> RAGStore:
    global _store
    if _store is None:
        _store = RAGStore()
    return _store


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    n = store().add_pdf(file.filename, data)
    if n == 0:
        raise HTTPException(422, "No extractable text found in the PDF.")
    return {"filename": file.filename, "chunks_indexed": n}


@app.post("/ask")
def ask(req: AskRequest):
    return store().answer(req.question, req.top_k)


@app.get("/documents")
def documents():
    s = store()
    names = sorted({c["source"] for c in s.chunks})
    return {"documents": names, "total_chunks": len(s.chunks)}
