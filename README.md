# CBU RAG Server

Local RAG pipeline for the CBU Lancer chatbot. Runs ChromaDB + Ollama on DGX.

---

## Setup (one time)

```bash
# 1. Install Ollama
brew install ollama
brew services start ollama

# 2. Pull models
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# 3. Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Every session

Open 3 terminal tabs in this directory.

**Tab 1 — ChromaDB**
```bash
source .venv/bin/activate
chroma run --host localhost --port 8001 --path ./chroma_data
```

**Tab 2 — RAG Server**
```bash
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 7860
```

**Tab 3 — TUI chat** (or your workspace)
```bash
source .venv/bin/activate
python tui_chat.py 2>/dev/null
```

The `2>/dev/null` suppresses Ollama's llama.cpp loader logs which would otherwise clutter the chat output.

Ollama runs in the background automatically via brew services — no tab needed.

---

## Ingesting content

```bash
# Single URL
python ingest.py --url https://www.calbaptist.edu/academics/

# All seed URLs (defined in ingest.py)
python ingest.py --seed

# PDF
python ingest.py --pdf ./docs/student_handbook.pdf

# Directory of .txt or .pdf files
python ingest.py --dir ./docs/
```

Re-ingesting the same URL is safe — it uses upsert so no duplicates.

---

## Testing

```bash
# Health check
curl http://localhost:7860/health

# Doc count
curl http://localhost:7860/count

# Chat
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What programs does CBU offer?"}'
```

---

## Resetting the vector DB

```bash
# Option 1: CLI flag (collection only, Chroma keeps running)
python ingest.py --reset

# Option 2: Nuke the data folder entirely (stop Chroma first)
rm -rf ./chroma_data
```

---

## Switching to DGX

In `server.py`, change:
```python
LLM_MODEL = "qwen2.5:1.5b"   →   LLM_MODEL = "llama3.1:8b"
CHROMA_HOST = "localhost"     →   CHROMA_HOST = "dgx.your-tailnet.ts.net"
```

---

## Connecting the Reachy Mini

In `server.py`, set:
```python
ROBOT_ENABLED = True
```

When enabled, the server runs in **voice mode** — it no longer waits for HTTP requests to drive conversation. Instead:

1. **Idle** — Lancer continuously tracks faces with its camera and rotates toward anyone speaking (DoA via the ReSpeaker mic array)
2. **Listening** — webrtcvad detects speech onset; the robot records until 1.5s of silence
3. **Thinking** — head tilts while the utterance runs through RAG
4. **Answering** — head returns to neutral and the robot speaks the response via its built-in TTS
5. **Confused** — antennas droop + slight head dip when it can't find relevant context

The HTTP `/chat` endpoint still works when the robot is connected (for the TUI or remote clients), but it shares the same conversation lock so voice and HTTP can't overlap.

**Firmware requirement:** `reachy-mini >= 1.5.1`. Verify your robot's firmware matches before connecting — mismatched versions can cause unexpected joint behavior.

---

## File overview

| File | What it does |
|---|---|
| `server.py` | FastAPI RAG server — retrieval + Ollama inference + robot control |
| `robot_actions.py` | `LancerRobot` — voice loop, face tracking, DoA, head poses, TTS |
| `ingest.py` | Scrapes/chunks/embeds CBU content into ChromaDB |
| `requirements.txt` | Python dependencies |
| `chroma_data/` | Persistent vector DB (git-ignored) |
