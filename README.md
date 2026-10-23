# RAG Document Q&A (FastAPI + LLM)

A backend service that lets you upload PDFs and ask questions about them. Answers are grounded in the documents and include source citations (file + page).

## How it works

1. **Ingest**: PDF text is extracted page by page with `pypdf`.
2. **Chunk**: `RecursiveCharacterTextSplitter` (LangChain) splits text into ~800-character chunks with overlap.
3. **Embed**: chunks are embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`), so no embedding API key is needed.
4. **Retrieve**: the query is embedded and the top-k chunks are found by cosine similarity.
5. **Generate**: retrieved chunks are sent to an LLM (Anthropic Claude API) with a prompt that restricts answers to the context and requires citations.

## Tech stack

Python, FastAPI, LangChain text splitters, sentence-transformers, NumPy, Anthropic API, Docker.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your_key_here
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/upload` | Upload a PDF (multipart `file`) |
| POST | `/ask` | `{"question": "...", "top_k": 4}` returns answer + sources |
| GET | `/documents` | List indexed documents |

```bash
curl -F "file=@notes.pdf" http://127.0.0.1:8000/upload
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## Docker

```bash
docker build -t rag-doc-qa .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key_here rag-doc-qa
```

## Limitations and next steps

- Vector index is in memory (lost on restart). Next: persist with Chroma/FAISS.
- Add LangGraph agent routing, retrieval evaluation, and a simple web UI.
