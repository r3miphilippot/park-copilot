"""Tool 5: semantic search (RAG) in the park guide written in ./knowledge."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from app.rag import index as rag_index
from app.rag.index import GuideHit
from app.tools.common import ToolError, tool_guard

# Calibrated on the real model: off-topic questions score up to ~0.35, but some relevant
# passages score that low too. So a low score only triggers a warning; the LLM decides.
WEAK_MATCH_SCORE = 0.40


class GuideSearch(BaseModel):
    query: str
    results: list[GuideHit]  # most relevant first
    note: str | None = None


@tool_guard
def search_park_guide(
    query: Annotated[str, Field(min_length=2, max_length=300)],
    k: Annotated[int, Field(ge=1, le=8)] = 4,
) -> GuideSearch | ToolError:
    """Search the park guide for practical advice: visit strategy, best times of day,
    rainy days and indoor rides, visiting with children, food, accessibility.

    The guide gives general advice only, never wait times. Quote it rather than inventing.

    Args:
        query: what you are looking for, IN FRENCH (the guide is written in French), e.g.
            "attractions intérieures" or "taille minimale enfants".
        k: number of passages to return (1-8).
    """
    hits = rag_index.get_guide_index().search(query, k)
    note = None
    if not hits or hits[0].score < WEAK_MATCH_SCORE:
        note = (
            "Low similarity: these passages may not answer the question. Use them only if they "
            "are clearly relevant, otherwise say the guide does not cover it."
        )
    return GuideSearch(query=query, results=hits, note=note)
