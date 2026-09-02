"""
Runtime configuration and conversation history for Lancer.

Settings live in a JSON file so the dashboard can change model, voice, retrieval
depth and thresholds without editing code or restarting the server. History is
appended to JSONL so the transcript survives restarts and can be reviewed later.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "lancer_config.json"
HISTORY_PATH = BASE / "lancer_history.jsonl"
HISTORY_LIMIT = 500          # entries returned to the dashboard

_lock = threading.Lock()

DEFAULTS: dict[str, Any] = {
    # generation
    "llm_model": "qwen2.5:3b",
    "temperature": 0.3,
    "max_sentences": 3,
    "history_turns": 6,
    # retrieval
    "top_k": 5,
    "dense_k": 20,
    "lexical_k": 20,
    "fuse_k": 20,
    "rerank_enabled": True,
    "verify_grounding": True,
    "relevance_min": -6.0,
    # speech
    "tts_voice": "bm_george",
    "tts_speed": 1.0,
    "whisper_model": "small.en",
    # robot behaviour (read by the app at startup)
    "speech_multiplier": 2.0,
    "speech_rms_ceiling": 0.045,
    "barge_enabled": True,
    "barge_echo_headroom": 1.4,
    "barge_grace": 0.8,
    "doa_enabled": True,
    "yolo_tracking": True,
    "aim_smoothing": 0.45,
    "aim_deadband_frac": 0.022,
    "doa_priority": True,
    "head_tracking_weight": 1.0,
    "persona": "",
}

_cache: dict[str, Any] | None = None
_cache_mtime: float = 0.0


def load() -> dict[str, Any]:
    """
    Return the current config, merging defaults over anything on disk.

    The file is re-read whenever it changes on disk. Caching it for the life of
    the process meant an edit from anywhere else was silently overwritten the
    next time this process saved.
    """
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            mtime = 0.0
        if _cache is not None and mtime != _cache_mtime:
            _cache = None
        _cache_mtime = mtime
        if _cache is None:
            data = {}
            if CONFIG_PATH.exists():
                try:
                    data = json.loads(CONFIG_PATH.read_text())
                except (OSError, json.JSONDecodeError):
                    data = {}
            _cache = {**DEFAULTS, **data}
        return dict(_cache)


def get(key: str) -> Any:
    return load().get(key, DEFAULTS.get(key))


def update(changes: dict[str, Any]) -> dict[str, Any]:
    """Apply changes, keeping only known keys, and persist them."""
    global _cache
    # Must read the file first: in a fresh process _cache is None, and building
    # from DEFAULTS alone silently discarded every setting already on disk.
    existing = load()
    with _lock:
        current = {**DEFAULTS, **existing}
        for key, value in changes.items():
            if key in DEFAULTS:
                current[key] = value
        CONFIG_PATH.write_text(json.dumps(current, indent=2))
        _cache = current
        try:
            globals()["_cache_mtime"] = CONFIG_PATH.stat().st_mtime
        except OSError:
            pass
        return dict(current)


def reset() -> dict[str, Any]:
    global _cache
    with _lock:
        _cache = dict(DEFAULTS)
        CONFIG_PATH.write_text(json.dumps(_cache, indent=2))
        return dict(_cache)


def record(entry: dict[str, Any]) -> None:
    """Append one exchange to the transcript."""
    entry = {"ts": time.time(), **entry}
    with _lock:
        with HISTORY_PATH.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


def history(limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with _lock:
        lines = HISTORY_PATH.read_text().splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear_history() -> None:
    with _lock:
        HISTORY_PATH.write_text("")
