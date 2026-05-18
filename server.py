"""
CBU Chatbot - RAG Server
Run: uvicorn server:app --host 0.0.0.0 --port 7860

Two modes depending on ROBOT_ENABLED:

  ROBOT_ENABLED = False (default)
    HTTP-only mode. The TUI and any web client POST to /chat and get a JSON response.
    No robot code is imported.

  ROBOT_ENABLED = True
    Voice mode. On startup, a background voice_loop() task listens via the robot's
    microphone, runs each utterance through RAG, and speaks the answer aloud.
    The /chat HTTP endpoint still works (for the TUI), but a _conversation_lock
    prevents voice and HTTP from running RAG simultaneously.
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
LLM_MODEL = "qwen2.5:1.5b"   # swap to llama3.1:8b on DGX
EMBED_MODEL = "nomic-embed-text"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
TOP_K = 5                     # number of ChromaDB chunks to retrieve per query
ROBOT_ENABLED = False         # set to True when the Reachy Mini is connected

robot = None
# Ensures the robot finishes speaking before starting the next TTS call.
_speak_lock = asyncio.Lock()
# Ensures voice loop and HTTP endpoint don't run RAG at the same time.
_conversation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start robot + voice loop on server boot; clean up on shutdown."""
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
    """Embed the query and fetch the top-K most similar chunks from ChromaDB."""
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
    """
    Core RAG logic shared by both the HTTP endpoint and the voice loop.
    Retrieves context from ChromaDB, triggers the thinking gesture, then
    calls Ollama with the system prompt + context + conversation history.
    Returns (answer_text, source_list).
    """
    context, sources = retrieve(query)
    if robot:
        robot.thinking()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Use this CBU info to answer:\n\n{context}"},
        *history[-6:],  # keep last 3 turns (6 messages) for conversational context
        {"role": "user", "content": query},
    ]
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"], sources


async def _respond_with_robot(output: str) -> None:
    """
    Trigger the appropriate robot reaction after a query is answered.
    If the LLM admitted it doesn't know, play the confused gesture (which
    also speaks the fallback phrase). Otherwise return to neutral and speak
    the answer — serialized through _speak_lock so concurrent calls queue up.
    """
    is_unknown = "don't have that specific information" in output
    if is_unknown:
        robot.confused()
    else:
        robot.answering()
        async with _speak_lock:
            await asyncio.to_thread(robot.speak, output)


async def voice_loop() -> None:
    """
    Background task that drives the robot in voice mode.

    Loop:
      - Idle: face tracking + DoA orientation threads are running
      - listen_for_question() blocks in a thread pool until speech is captured and transcribed
      - Idle behaviors are paused for the duration of the conversation
      - RAG runs, robot reacts, then idle behaviors restart
    """
    robot.start_idle_behaviors()
    while True:
        # Runs in a thread pool — blocks without holding the event loop.
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
    """
    HTTP chat endpoint — used by the TUI and any web client.
    Acquires _conversation_lock so it can't overlap with the voice loop.
    """
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
