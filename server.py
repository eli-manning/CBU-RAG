"""
CBU Chatbot - RAG Server
Run: uvicorn server:app --host 0.0.0.0 --port 7860
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
LLM_MODEL = "qwen2.5:1.5b"       # swap to llama3.1:8b on DGX
EMBED_MODEL = "nomic-embed-text"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
TOP_K = 5
ROBOT_ENABLED = False  # Set to True if robot is connected

robot = None
_speak_lock = asyncio.Lock()
_conversation_lock = asyncio.Lock()  # prevents voice and HTTP from overlapping


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot
    if ROBOT_ENABLED:
        from robot_actions import LancerRobot
        robot = LancerRobot()
        robot.greet()
        asyncio.create_task(voice_loop())
    yield
    if robot:
        robot.stop_idle_behaviors()


app = FastAPI(title="CBU RAG Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = """You are Lancer, CBU's ACM AI. Use the provided context to answer questions.

STRICT CONSTRAINTS:
- Answer in EXACTLY 2 to 3 sentences.
- Use plain text ONLY. No bolding (**), no headers (###), and no lists.
- If you don't know the answer based on the context, say: "I'm sorry, I don't have that specific information in my current database."
- Focus only on the specific question asked."""

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


async def _process_query(query: str, history: list[dict] = []) -> tuple[str, list[str]]:
    """Shared RAG logic used by both the HTTP endpoint and the voice loop."""
    context, sources = retrieve(query)
    if robot:
        robot.thinking()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Use this CBU info to answer:\n\n{context}"},
        *history[-6:],
        {"role": "user", "content": query},
    ]
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"], sources


async def _respond_with_robot(output: str) -> None:
    """Handle robot reactions and blocking TTS after a query is answered."""
    is_unknown = "don't have that specific information" in output
    if is_unknown:
        robot.confused()
    else:
        robot.answering()
        async with _speak_lock:
            await asyncio.to_thread(robot.speak, output)


async def voice_loop() -> None:
    """Continuous listen → RAG → speak loop, runs as a background task when ROBOT_ENABLED."""
    robot.start_idle_behaviors()
    while True:
        question = await asyncio.to_thread(robot.listen_for_question)
        if not question:
            continue

        logger.info(f"[voice] heard: {question!r}")

        async with _conversation_lock:
            robot.stop_idle_behaviors()
            try:
                output, _ = await _process_query(question)
                await _respond_with_robot(output)
            except Exception as e:
                logger.error(f"[voice] error: {e}")
            finally:
                robot.start_idle_behaviors()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        async with _conversation_lock:
            if robot:
                robot.stop_idle_behaviors()
            try:
                output, sources = await _process_query(req.query, req.conversation_history)
                if robot:
                    await _respond_with_robot(output)
            finally:
                if robot:
                    robot.start_idle_behaviors()

        return ChatResponse(answer=output, sources=sources, model_used=LLM_MODEL)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "docs": collection.count()}


@app.get("/count")
async def count():
    return {"docs": collection.count()}


@app.post("/greet")
async def greet():
    if robot:
        robot.greet()
    return {"status": "greeted"}
