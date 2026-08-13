"""
llm_backends package

Pluggable LLM providers behind one common interface (see base.py). Add a
new provider by creating a new_backend.py implementing LLMBackend, then
registering it in PROVIDERS below.
"""

from src.llm_backends.base import LLMBackend
from src.llm_backends.gemini_backend import GeminiBackend
from src.llm_backends.ollama_backend import OllamaBackend
from src.llm_backends.openai_backend import OpenAIBackend

PROVIDERS = {
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
    "ollama": OllamaBackend,
}


def get_llm_backend(provider: str, model: str | None = None, **kwargs) -> LLMBackend:
    """Factory: build an LLMBackend for the given provider name.

    Args:
        provider: One of "openai", "gemini", "ollama".
        model: Model name/string for that provider. If omitted, each
            backend falls back to its own sensible default.
        **kwargs: Passed through to the backend constructor (e.g.
            api_key, host).

    Returns:
        An instance implementing LLMBackend.
    """
    key = provider.lower().strip()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}"
        )
    backend_cls = PROVIDERS[key]
    if model:
        return backend_cls(model=model, **kwargs)
    return backend_cls(**kwargs)


__all__ = [
    "LLMBackend",
    "OpenAIBackend",
    "GeminiBackend",
    "OllamaBackend",
    "get_llm_backend",
]
