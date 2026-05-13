"""
CBU Chatbot - RAG Server
Run: uvicorn server:app --host 0.0.0.0 --port 7860
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CBU RAG Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Config ---
LLM_MODEL = "qwen2.5:1.5b"       # swap to llama3.1:8b on DGX
EMBED_MODEL = "nomic-embed-text"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
TOP_K = 5

SYSTEM_PROMPT = """You are Lancer, CBU's friendly AI assistant. Help students, faculty,
and visitors with questions about California Baptist University — programs, campus life,
events, policies, and more. Be concise and warm. Answer in 2-4 sentences.
If you don't know something, say so honestly rather than guessing."""

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(
    name="cbu_knowledge",
    metadata={"hnsw:space": "cosine"}
)


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    model_used: str


def retrieve(query: str) -> tuple[str, list[str]]:
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    docs = results["documents"][0]
    sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
    return "\n\n---\n\n".join(docs), sources


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        context, sources = retrieve(req.query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Use this CBU info to answer:\n\n{context}"},
            *req.conversation_history[-6:],
            {"role": "user", "content": req.query},
        ]
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        return ChatResponse(
            answer=response["message"]["content"],
            sources=sources,
            model_used=LLM_MODEL
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "docs": collection.count()}


@app.get("/count")
async def count():
    return {"docs": collection.count()}
