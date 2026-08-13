"""
ollama_backend.py

LLM backend that calls a locally running Ollama server. No API key or
internet connection needed once the model is pulled.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3
    3. Ollama runs a local server at http://localhost:11434 automatically.

Requires: pip install requests (usually already present).
"""

import requests

from src.llm_backends.base import LLMBackend

DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` "
                "running, and have you pulled the model with "
                f"`ollama pull {self.model}`?"
            ) from e

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content.strip()
