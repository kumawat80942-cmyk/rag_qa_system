"""
base.py

Common interface every LLM backend implements. The rest of the app
(generator.py, app.py, cli.py) only ever talks to this interface, so
swapping providers never touches retrieval or UI code.
"""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a response given a system prompt and a user message.

        Args:
            system_prompt: Instructions + retrieved context.
            user_message: The user's question.

        Returns:
            The model's answer as plain text.
        """
        raise NotImplementedError
