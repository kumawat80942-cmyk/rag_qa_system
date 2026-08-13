"""
generator.py

Composes an answer from retrieved passages using whichever LLM backend
is configured (Gemini, OpenAI, or a local Ollama model — see
src/llm_backends/). This is the "G" in RAG: it's instructed to answer
only from the supplied context, not from the model's own knowledge.
"""

from src.llm_backends import LLMBackend, get_llm_backend
from src.retriever import RetrievedPassage

SYSTEM_PROMPT_TEMPLATE = """You are a document question-answering assistant. \
Answer the user's question using ONLY the excerpts provided below. Cite \
which excerpt number(s) support your answer, like (Excerpt 1). If the \
excerpts do not contain enough information to answer, say so directly \
instead of guessing.

{context}
"""


class AnswerGenerator:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        provider: str = "gemini",
        model: str | None = None,
        **backend_kwargs,
    ):
        """
        Args:
            backend: A ready-made LLMBackend instance. If provided,
                `provider`/`model` are ignored.
            provider: "gemini", "openai", or "ollama" — used to build a
                backend if one wasn't passed directly.
            model: Model name for that provider (optional, each backend
                has its own default).
            **backend_kwargs: Extra args for the backend constructor,
                e.g. api_key="...", host="http://localhost:11434".
        """
        self.backend = backend or get_llm_backend(provider, model=model, **backend_kwargs)

    @staticmethod
    def _build_context(passages: list[RetrievedPassage]) -> str:
        blocks = []
        for i, p in enumerate(passages, start=1):
            blocks.append(
                f"Excerpt {i} (passage #{p.chunk.index + 1}):\n{p.chunk.text}"
            )
        return "\n\n".join(blocks)

    def generate_answer(self, question: str, passages: list[RetrievedPassage]) -> str:
        """Generate an answer to `question` grounded in `passages`."""
        if not passages:
            return (
                "No relevant passages were found in the document for this "
                "question, so I can't answer it from the source text."
            )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=self._build_context(passages)
        )
        answer = self.backend.generate(system_prompt, question)
        return answer or "No answer was returned."
