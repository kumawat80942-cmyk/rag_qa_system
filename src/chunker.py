"""
chunker.py

Splits raw document text into overlapping chunks suitable for indexing
and retrieval. Chunking by sentence groups (rather than fixed character
windows) keeps each chunk semantically coherent.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation + capitalization cues."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(normalized)
    return [s.strip() for s in sentences if s.strip()]


def build_chunks(
    text: str,
    chunk_size: int = 3,
    chunk_overlap: int = 1,
    min_chunk_chars: int = 15,
) -> list[Chunk]:
    """Group sentences into overlapping chunks.

    Args:
        text: Raw document text.
        chunk_size: Number of sentences per chunk.
        chunk_overlap: Number of sentences shared between consecutive
            chunks (must be smaller than chunk_size).
        min_chunk_chars: Minimum chunk length to keep (filters out
            near-empty trailing chunks).

    Returns:
        A list of Chunk objects, each with an index and its text.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        window = sentences[i : i + chunk_size]
        if not window:
            break
        chunk_text = " ".join(window)
        if len(chunk_text) >= min_chunk_chars:
            chunks.append(Chunk(index=len(chunks), text=chunk_text))
        if i + chunk_size >= len(sentences):
            break
        i += step

    if not chunks:
        chunks = [Chunk(index=0, text=text.strip())]

    return chunks
