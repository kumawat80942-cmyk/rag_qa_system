"""
gemini_backend.py

LLM backend that calls the Google Gemini API.
Requires: pip install google-genai, and GEMINI_API_KEY set.
"""

import os

from src.llm_backends.base import LLMBackend

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_tokens: int = 1000):
        self.genai_mod = None
        self.genai_legacy = None
        self.backend_type = None

        try:
            from google import genai
            self.genai_mod = genai
            self.backend_type = "genai"
        except ImportError:
            try:
                import google.generativeai as genai_legacy
                self.genai_legacy = genai_legacy
                self.backend_type = "generativeai"
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for GeminiBackend. "
                    "Install it with `pip install google-genai`."
                ) from e

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in your "
                "environment or .env file."
            )

        self.model = model
        self.max_tokens = max_tokens

        if self.backend_type == "genai":
            self.client = self.genai_mod.Client(api_key=key)
        else:
            self.genai_legacy.configure(api_key=key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        if self.backend_type == "genai":
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens,
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=config,
            )
            return (response.text or "").strip()
        else:
            model_obj = self.genai_legacy.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
            response = model_obj.generate_content(
                user_message,
                generation_config={"max_output_tokens": self.max_tokens},
            )
            return (response.text or "").strip()
