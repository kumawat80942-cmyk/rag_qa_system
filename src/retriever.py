"""
retriever.py

TF-IDF based retrieval over document chunks. Builds a sparse TF-IDF
index with scikit-learn and ranks chunks against a query using cosine
similarity.

This is the "R" in RAG: it decides which passages are relevant enough
to hand to the generator.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import Chunk


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float


class TfidfRetriever:
    """Fits a TF-IDF index over a list of chunks and retrieves the
    top-k most similar chunks for a given query."""

    def __init__(self, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.chunks: list[Chunk] = []
        self._matrix = None

    def fit(self, chunks: list[Chunk]) -> None:
        """Build the TF-IDF index over the given chunks."""
        if not chunks:
            raise ValueError("Cannot fit retriever on an empty chunk list")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    @property
    def vocabulary_size(self) -> int:
        if self._matrix is None:
            return 0
        return len(self.vectorizer.vocabulary_)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        """Return the top_k chunks most similar to the query, ranked by
        cosine similarity. Chunks with zero similarity are excluded."""
        if self._matrix is None:
            raise RuntimeError("Call fit() before retrieve()")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        results = [
            RetrievedPassage(chunk=chunk, score=float(score))
            for chunk, score in ranked
            if score > 0
        ]
        return results[:top_k]
