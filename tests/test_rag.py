import hashlib
import math
import os
import re

import pytest

from app.config import get_settings
from app.rag import index as rag_index
from app.rag.chunking import load_knowledge, split_markdown
from app.rag.index import GuideIndex
from app.tools import ToolError, search_park_guide

DOC = """# Jours de pluie

Intro sous le titre.

## Attractions intérieures

### Disneyland Park
- Pirates of the Caribbean

### Adventure World
- Ratatouille

## Ce qu'il faut prévoir
Un poncho.
"""


class FakeEmbedder:
    """Deterministic bag-of-words vectors: tests run without downloading the real model."""

    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# --------------------------------------------------------------------------- chunking


def test_split_markdown_follows_heading_hierarchy():
    chunks = split_markdown(DOC, "pluie.md")
    assert [(c.title, c.section, c.text) for c in chunks] == [
        ("Jours de pluie", "Introduction", "Intro sous le titre."),
        (
            "Jours de pluie",
            "Attractions intérieures > Disneyland Park",
            "- Pirates of the Caribbean",
        ),
        ("Jours de pluie", "Attractions intérieures > Adventure World", "- Ratatouille"),
        ("Jours de pluie", "Ce qu'il faut prévoir", "Un poncho."),
    ]  # the empty "Attractions intérieures" body produces no chunk
    assert chunks[0].id == "pluie.md#0"
    assert chunks[1].embedding_text().startswith("Jours de pluie — Attractions intérieures")


def test_long_section_is_split_on_paragraphs():
    paragraphs = [f"Paragraphe {i} " + "x" * 80 for i in range(10)]
    chunks = split_markdown("## Long\n\n" + "\n\n".join(paragraphs), "long.md", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.section == "Long" for c in chunks)
    assert "\n\n".join(c.text for c in chunks) == "\n\n".join(paragraphs)  # nothing lost


def test_real_knowledge_base_loads():
    chunks = load_knowledge(get_settings().knowledge_dir)
    assert {c.source for c in chunks} == {
        "accessibilite.md", "jours-de-pluie.md", "meilleurs-moments.md",
        "restauration.md", "strategie-de-visite.md", "visite-avec-enfants.md",
    }  # fmt: skip
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique


# --------------------------------------------------------------------------- index + tool


@pytest.fixture
def fake_index(monkeypatch):
    index = GuideIndex(split_markdown(DOC, "pluie.md"), FakeEmbedder())
    monkeypatch.setattr(rag_index, "get_guide_index", lambda: index)
    return index


def test_index_returns_best_section_first(fake_index):
    hits = fake_index.search("ratatouille", k=2)
    assert hits[0].section == "Attractions intérieures > Adventure World"
    assert hits[0].source == "pluie.md" and 0 < hits[0].score <= 1
    assert hits[0].score >= hits[1].score


def test_k_larger_than_index_is_fine(fake_index):
    assert len(fake_index.search("pluie", k=50)) == 4


def test_search_tool(fake_index):
    result = search_park_guide("poncho prévoir", k=1)
    assert result.results[0].section == "Ce qu'il faut prévoir"
    assert result.note is None


def test_search_tool_flags_weak_matches(fake_index):
    result = search_park_guide("zzz inconnu")
    assert "Low similarity" in result.note


def test_search_tool_validates_query(fake_index):
    assert isinstance(search_park_guide(""), ToolError)
    assert isinstance(search_park_guide("pluie", k=0), ToolError)


def test_search_tool_reports_index_failure(monkeypatch):
    def broken():
        raise ValueError("the knowledge base is empty")

    monkeypatch.setattr(rag_index, "get_guide_index", broken)
    assert isinstance(search_park_guide("pluie"), ToolError)


# --------------------------------------------------------------------------- real model


@pytest.mark.integration
@pytest.mark.skipif(bool(os.getenv("CI")), reason="downloads the 220 MB model; run locally")
def test_real_model_finds_the_right_guide():
    index = rag_index.get_guide_index()
    cases = {
        "que faire quand il pleut ?": "jours-de-pluie.md",
        "attractions pour un enfant de 4 ans": "visite-avec-enfants.md",
        "fauteuil roulant": "accessibilite.md",
        "c'est quoi une file single rider": "strategie-de-visite.md",
        "où manger le midi": "restauration.md",
    }
    for query, source in cases.items():
        assert index.search(query, k=1)[0].source == source, query
