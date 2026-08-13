"""
openai_backend.py

LLM backend that calls the OpenAI Chat Completions API.
Requires: pip install openai, and OPENAI_API_KEY set.
"""

import os

from src.llm_backends.base import LLMBackend

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_tokens: int = 1000):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with `pip install openai`."
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY in your "
                "environment or .env file."
            )
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
