"""Minimal RAG pipeline: PDF -> chunks -> embeddings -> cosine retrieval -> LLM answer."""
import io
import os

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import anthropic

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context excerpts. "
    "Cite sources as [source, page N]. If the context does not contain the answer, "
    "say you could not find it in the uploaded documents."
)


class RAGStore:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def add_pdf(self, filename: str, data: bytes) -> int:
        reader = PdfReader(io.BytesIO(data))
        new_chunks = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for piece in self.splitter.split_text(text):
                new_chunks.append({"text": piece, "source": filename, "page": page_no})
        if not new_chunks:
            return 0
        vecs = self.embedder.encode(
            [c["text"] for c in new_chunks], normalize_embeddings=True
        )
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.embeddings is None or not self.chunks:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def answer(self, question: str, k: int = 4) -> dict:
        hits = self.search(question, k)
        if not hits:
            return {"answer": "No documents uploaded yet.", "sources": []}
        context = "\n\n".join(
            f"[{h['source']}, page {h['page']}]\n{h['text']}" for h in hits
        )
        msg = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        sources = [
            {"source": h["source"], "page": h["page"], "score": round(h["score"], 3)}
            for h in hits
        ]
        return {"answer": text, "sources": sources}
