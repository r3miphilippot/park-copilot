"""Vector index of the park guide: FastEmbed embeddings stored in an in-memory Chroma.

The guide is small (a few dozen chunks), so the index is simply rebuilt at startup:
no persistent volume is needed on Hugging Face Spaces.
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from app.config import get_settings
from app.rag.chunking import Chunk, load_knowledge

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Runs the ONNX embedding model locally on CPU: free, no API key, no rate limit."""

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # heavy import, only when really needed

        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir or ""))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class GuideHit(BaseModel):
    source: str
    title: str
    section: str
    text: str
    score: float  # cosine similarity, 1 = identical meaning


class GuideIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("the knowledge base is empty")
        self.embedder = embedder
        client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        # Ephemeral clients share one in-process store: a unique name keeps indexes separate.
        self.collection = client.create_collection(
            name=f"guide_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we pass our own FastEmbed vectors
        )
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embedder.embed_documents([c.embedding_text() for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "title": c.title, "section": c.section} for c in chunks
            ],
        )
        self.size = len(chunks)

    @classmethod
    def from_directory(cls, directory: Path, embedder: Embedder) -> GuideIndex:
        started = time.perf_counter()
        index = cls(load_knowledge(directory), embedder)
        log.info("guide index built: %d chunks in %.1fs", index.size, time.perf_counter() - started)
        return index

    def search(self, query: str, k: int = 4) -> list[GuideHit]:
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, self.size),
            include=["documents", "metadatas", "distances"],
        )
        return [
            GuideHit(**meta, text=doc, score=round(1 - distance, 3))
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]


@lru_cache
def get_guide_index() -> GuideIndex:
    """Built once per process (at API startup, or lazily on first search)."""
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
    return GuideIndex.from_directory(settings.knowledge_dir, embedder)
