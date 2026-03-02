#!/usr/bin/env python3
"""OranAI (oran-a1) local inference server.

- No ChatGPT/Gemini API usage.
- Uses PyTorch + Transformers for local neural text generation.
- Provides chat history + multiple chats persisted in SQLite.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from flask import Flask, jsonify, request, send_from_directory
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: flask. Install dependencies with `python -m pip install -r requirements.txt`."
    ) from exc


@dataclass
class InferenceConfig:
    model_name: str = os.getenv("ORAN_A1_MODEL", "distilgpt2")
    max_new_tokens: int = int(os.getenv("ORAN_A1_MAX_NEW_TOKENS", "120"))
    temperature: float = float(os.getenv("ORAN_A1_TEMPERATURE", "0.8"))
    top_p: float = float(os.getenv("ORAN_A1_TOP_P", "0.92"))


class OranA1Engine:
    """Local deep-learning text generation engine backed by PyTorch/Transformers."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._lock = Lock()
        self._ready = False
        self._error: str | None = None
        self._tokenizer = None
        self._model = None
        self._torch = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        """Load model lazily, keeping startup quick and resilient."""
        if self._ready or self._error:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            model.to(device)
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._ready = True
        except Exception as exc:  # graceful runtime failure for missing deps/model
            self._error = (
                "OranAI model failed to initialize. Install dependencies and ensure model download access. "
                f"Details: {exc}"
            )

    def _build_prompt(self, history: list[dict[str, str]], user_prompt: str, tone: str = "balanced") -> str:
        tone_instruction = {
            "balanced": "Keep answers clear and practical.",
            "concise": "Keep answers short and to the point.",
            "detailed": "Provide a thorough explanation with structured steps.",
        }.get(tone, "Keep answers clear and practical.")
        system = (
            "You are oran-a1, a helpful assistant made by TheGeneric. "
            f"{tone_instruction}"
        )
        chunks = [f"System: {system}"]
        for item in history[-10:]:
            role = item.get("role", "user").strip().lower()
            content = item.get("content", "").strip()
            if content:
                chunks.append(f"{'Assistant' if role == 'assistant' else 'User'}: {content}")
        chunks.append(f"User: {user_prompt.strip()}")
        chunks.append("Assistant:")
        return "\n".join(chunks)

    def _fallback_response(self, history: list[dict[str, str]], user_prompt: str, tone: str) -> str:
        cleaned = " ".join(user_prompt.split())
        recent = [m.get("content", "").strip() for m in history[-2:] if m.get("content")]
        context = f"\nRecent context: {' | '.join(recent)}" if recent else ""
        if tone == "concise":
            return f"I cannot load the neural model right now, but I can still help. Main answer: {cleaned[:280]}.{context}".strip()
        if tone == "detailed":
            return (
                "Neural model is temporarily unavailable, so I am using a local fallback mode.\n"
                f"Prompt received: {cleaned}\n"
                "Suggested next steps:\n"
                "1) Re-run with network/model access enabled for full deep-learning output.\n"
                "2) If this is a coding task, share files or error logs and I will provide targeted fixes.\n"
                "3) If this is a planning task, I can still draft a complete step-by-step plan now."
                f"{context}"
            )
        return (
            "Neural model is currently unavailable, so this is a local fallback answer. "
            f"I understood your prompt as: {cleaned}. "
            "I can still provide practical guidance while model dependencies are loading."
            f"{context}"
        )

    def generate(self, history: list[dict[str, str]], user_prompt: str, tone: str = "balanced") -> dict[str, Any]:
        self.load()
        started = time.perf_counter()

        if not self._ready:
            answer = self._fallback_response(history=history, user_prompt=user_prompt, tone=tone)
            return {
                "ok": True,
                "answer": answer,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "error": self._error or "Fallback mode",
            }

        with self._lock:
            prompt = self._build_prompt(history, user_prompt, tone=tone)
            tok = self._tokenizer
            model = self._model
            torch = self._torch

            inputs = tok(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=tok.eos_token_id,
                )

            decoded = tok.decode(output[0], skip_special_tokens=True)
            text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
            answer = text.split("\nUser:")[0].strip() or "I need a bit more detail to answer well."

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {"ok": True, "answer": answer, "elapsed_ms": elapsed_ms, "error": None}


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                  chat_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  elapsed_ms INTEGER,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
                );
                """
            )

    def create_chat(self, title: str = "New Chat") -> str:
        chat_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chats(chat_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title, now, now),
            )
        return chat_id

    def list_chats(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chat_id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, elapsed_ms, created_at FROM messages WHERE chat_id=? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_message(self, chat_id: str, role: str, content: str, elapsed_ms: int | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(chat_id, role, content, elapsed_ms, created_at) VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content, elapsed_ms, now),
            )
            conn.execute("UPDATE chats SET updated_at=? WHERE chat_id=?", (now, chat_id))


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".oranai"
STORE = ChatStore(DATA_DIR / "oranai.sqlite3")
ENGINE = OranA1Engine()
APP = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


@APP.get("/")
def home() -> Any:
    return send_from_directory(BASE_DIR, "index.html")


@APP.get("/oranai")
def oranai_page() -> Any:
    return send_from_directory(BASE_DIR, "oranai.html")


@APP.get("/api/health")
def health() -> Any:
    ENGINE.load()
    return jsonify({"ready": ENGINE.ready, "error": ENGINE.error, "model": ENGINE.config.model_name})


@APP.post("/api/chats")
def create_chat() -> Any:
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "New Chat").strip()[:80]
    chat_id = STORE.create_chat(title=title)
    return jsonify({"chat_id": chat_id})


@APP.get("/api/chats")
def chats() -> Any:
    return jsonify({"chats": STORE.list_chats()})


@APP.get("/api/chats/<chat_id>")
def chat_messages(chat_id: str) -> Any:
    return jsonify({"messages": STORE.get_messages(chat_id)})


@APP.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(force=True)
    prompt = (payload.get("prompt") or "").strip()
    chat_id = (payload.get("chat_id") or "").strip()
    tone = (payload.get("tone") or "balanced").strip().lower()
    if tone not in {"balanced", "concise", "detailed"}:
        tone = "balanced"

    if not prompt:
        return jsonify({"ok": False, "error": "Prompt is required."}), 400

    if not chat_id:
        chat_id = STORE.create_chat(title=prompt[:32] or "New Chat")

    existing = STORE.get_messages(chat_id)
    history = [{"role": m["role"], "content": m["content"]} for m in existing]
    STORE.add_message(chat_id, "user", prompt)
    result = ENGINE.generate(history=history, user_prompt=prompt, tone=tone)

    if result["ok"]:
        STORE.add_message(chat_id, "assistant", result["answer"], elapsed_ms=result["elapsed_ms"])

    return jsonify({"chat_id": chat_id, **result})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
