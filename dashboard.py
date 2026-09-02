"""
Lancer dashboard: configuration, transcript, retrieval inspection, robot control.

Mounted by server.py. Everything here is read/write against the same runtime
config the voice pipeline uses, so changes apply to the next question without a
restart -- except the Whisper model, which is reloaded on demand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import lancer_runtime as rt

router = APIRouter()

# Same failover as the app: wired link when present, Tailscale otherwise.
ROBOT_HOSTS = [h for h in (os.environ.get("LANCER_ROBOT"),
                           "10.5.1.1", "100.116.136.5") if h]
ROBOT_HOST = ROBOT_HOSTS[0]


def robot_api() -> str:
    """Base URL of the first reachable robot daemon."""
    global ROBOT_HOST
    for host in ROBOT_HOSTS:
        try:
            if requests.get(f"http://{host}:8000/api/daemon/status", timeout=3).ok:
                ROBOT_HOST = host
                return f"http://{host}:8000"
        except requests.RequestException:
            continue
    return f"http://{ROBOT_HOSTS[0]}:8000"
APP_NAME = "reachy_mini_lancer_app"
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


class ConfigUpdate(BaseModel):
    changes: dict[str, Any]


class QueryRequest(BaseModel):
    query: str


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    if not DASHBOARD_HTML.exists():
        return "<h1>dashboard.html missing</h1>"
    return DASHBOARD_HTML.read_text()


@router.get("/api/config")
async def get_config() -> dict:
    return {"config": rt.load(), "defaults": rt.DEFAULTS}


@router.post("/api/config")
async def set_config(update: ConfigUpdate) -> dict:
    return {"config": rt.update(update.changes)}


@router.post("/api/config/reset")
async def reset_config() -> dict:
    return {"config": rt.reset()}


@router.get("/api/history")
async def get_history(limit: int = 200) -> dict:
    return {"history": rt.history(limit)}


@router.delete("/api/history")
async def delete_history() -> dict:
    rt.clear_history()
    return {"status": "ok"}


@router.get("/api/models")
async def list_models() -> dict:
    """Ollama models available for generation (embedding models filtered out)."""
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in res.json().get("models", [])]
        return {"models": [n for n in names if "embed" not in n]}
    except requests.RequestException as e:
        return {"models": [], "error": str(e)}


@router.get("/api/voices")
async def list_voices() -> dict:
    """Kokoro voices, grouped so the picker is navigable."""
    try:
        import server

        voices = sorted(server._get_kokoro().get_voices())
    except Exception as e:
        return {"voices": [], "error": str(e)}
    return {"voices": voices}


@router.get("/api/robot/status")
async def robot_status() -> dict:
    out: dict[str, Any] = {"host": ROBOT_HOST}
    try:
        daemon = requests.get(f"{robot_api()}/api/daemon/status", timeout=4).json()
        out["online"] = True
        out["state"] = daemon.get("state")
        out["version"] = daemon.get("version")
        out["wlan_ip"] = daemon.get("wlan_ip")
        backend = daemon.get("backend_status") or {}
        out["motors"] = backend.get("motor_control_mode")
        stats = backend.get("control_loop_stats") or {}
        out["control_hz"] = round(stats.get("mean_control_loop_frequency", 0), 1)
    except requests.RequestException as e:
        out["online"] = False
        out["error"] = str(e)
        return out
    try:
        app = requests.get(f"{robot_api()}/api/apps/current-app-status", timeout=4).json()
        out["app"] = (app or {}).get("info", {}).get("name")
        out["app_state"] = (app or {}).get("state")
    except requests.RequestException:
        out["app"] = None
    try:
        vol = requests.get(f"{robot_api()}/api/volume/current", timeout=4).json()
        out["volume"] = vol.get("volume")
    except requests.RequestException:
        pass
    return out


@router.post("/api/robot/{action}")
async def robot_control(action: str, volume: int | None = None) -> dict:
    """start / stop the Lancer app, or set speaker volume."""
    try:
        if action == "start":
            r = requests.post(f"{robot_api()}/api/apps/start-app/{APP_NAME}", timeout=60)
        elif action == "stop":
            r = requests.post(f"{robot_api()}/api/apps/stop-current-app", timeout=30)
        elif action == "volume" and volume is not None:
            r = requests.post(f"{robot_api()}/api/volume/set",
                              json={"volume": int(volume)}, timeout=10)
        elif action == "test-sound":
            r = requests.post(f"{robot_api()}/api/volume/test-sound", timeout=20)
        else:
            raise HTTPException(status_code=400, detail=f"unknown action {action}")
        return {"status": "ok", "code": r.status_code}
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/api/retrieve")
async def inspect_retrieval(req: QueryRequest) -> dict:
    """Run retrieval only, so chunk quality can be judged without generating."""
    import asyncio

    import server

    context, sources, relevant, score = await asyncio.to_thread(
        server.retrieve, req.query
    )
    chunks = context.split("\n\n---\n\n") if context else []
    return {
        "query": req.query,
        "relevant": relevant,
        "best_score": score,
        "results": [
            {"source": Path(s).name, "text": c.strip()[:600]}
            for s, c in zip(sources, chunks)
        ],
    }


@router.post("/api/ask")
async def ask(req: QueryRequest) -> dict:
    """Text chat through the same RAG path the voice pipeline uses."""
    import server

    answer, sources = await server._process_query(req.query, server._history)
    answer = server.sanitize_for_speech(answer)
    server._history.append({"role": "user", "content": req.query})
    server._history.append({"role": "assistant", "content": answer})
    del server._history[:-int(rt.get("history_turns"))]
    rt.record({"heard": req.query, "answer": answer, "sources": sources,
               "unknown": server.NO_ANSWER in answer, "model": rt.get("llm_model"),
               "via": "dashboard"})
    return {"answer": answer, "sources": [Path(s).name for s in sources]}


@router.get("/api/stats")
async def stats() -> dict:
    import server

    return {
        "chunks": server.collection.count(),
        "lexical_chunks": len(server._bm25_docs),
        "history_entries": len(rt.history(10_000)),
        "reranker_loaded": server._reranker is not None,
        "whisper_loaded": server._whisper_model is not None,
        "kokoro_loaded": server._kokoro is not None,
    }
