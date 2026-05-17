# CBU RAG Server

Local RAG pipeline for the CBU Lancer chatbot. Runs ChromaDB + Ollama on your Mac during development, swaps to DGX for production.

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

**Tab 3 — Your workspace** (ingest, test, etc.)
```bash
source .venv/bin/activate
```

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

On startup, Lancer will greet automatically. During chat:
- **Thinking** — head tilts right while the RAG query runs
- **Answering** — head returns to neutral, robot speaks the response aloud
- **Confused** — antennas droop + slight head dip when it can't find context

Concurrent requests are serialized through a lock so the robot finishes speaking before starting the next response.

**Firmware requirement:** `reachy-mini >= 1.5.1`. Verify your robot's firmware matches before connecting — mismatched versions can cause unexpected joint behavior.

---

## File overview

| File | What it does |
|---|---|
| `server.py` | FastAPI RAG server — retrieval + Ollama inference + robot control |
| `robot_actions.py` | `LancerRobot` wrapper — head poses, antenna animations, TTS |
| `ingest.py` | Scrapes/chunks/embeds CBU content into ChromaDB |
| `requirements.txt` | Python dependencies |
| `chroma_data/` | Persistent vector DB (git-ignored) |
