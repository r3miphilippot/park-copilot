"""Split the markdown guide into one chunk per section.

Sections are natural units of meaning in a guide ("Jours de pluie > Ce qu'il faut prévoir"),
so they make better chunks than fixed-size windows. A very long section is further split
on paragraph boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1200
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name, e.g. "jours-de-pluie.md"
    title: str  # document title (h1)
    section: str  # "h2 > h3" path, or "Introduction" for text right under the title
    text: str

    def embedding_text(self) -> str:
        """Titles are embedded with the body: a query about "pluie" must match a bullet list
        that never repeats the word."""
        return f"{self.title} — {self.section}\n{self.text}"


def split_markdown(markdown: str, source: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    title = Path(source).stem
    path: list[str] = []  # current h2/h3 headings
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        for part in _split_long(body, max_chars) if body else []:
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks)}",
                    source=source,
                    title=title,
                    section=" > ".join(path) or "Introduction",
                    text=part,
                )
            )

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()  # a heading closes the previous section
        level, name = len(match.group(1)), match.group(2).strip()
        if level == 1:
            title, path = name, []
        else:
            path = [*path[: level - 2], name]  # h2 resets the path, h3 nests under h2
    flush()
    return chunks


def _split_long(body: str, max_chars: int) -> list[str]:
    """Group paragraphs into parts of at most `max_chars` (a single huge paragraph stays whole)."""
    if len(body) <= max_chars:
        return [body]
    parts, current = [], ""
    for paragraph in re.split(r"\n\s*\n", body):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def load_knowledge(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(directory.glob("*.md")):
        chunks.extend(split_markdown(file.read_text(encoding="utf-8"), file.name))
    return chunks
