"""Deterministic checks on an agent answer (no LLM involved, unit-tested).

Each check returns a Check(category, name, passed, detail). Categories are what the report
aggregates: mode, tool_choice, no_hallucination, transparency, content, language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Check:
    category: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Transcript:
    """What one eval case produced."""

    answer: str
    mode: str | None
    target_date: str | None
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)  # raw JSON of every ToolMessage

    @property
    def tools_called(self) -> set[str]:
        return {name for name, _ in self.tool_calls}


# --------------------------------------------------------------------------- helpers

# "25 min", "20-30 minutes", "15 à 20 min", "≈ 22,5 min", "10 mins"
WAIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)(?:\s*(?:-|–|à|to|/)\s*(\d+(?:[.,]\d+)?))?\s*(?:min(?:ute)?s?)\b",
    re.IGNORECASE,
)
DAYS_RE = re.compile(r"(\d+)\s*(?:jours?|days?)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

NO_HISTORY_WORDS = [
    "pas encore", "aucune donnée", "aucun historique", "pas d'historique", "pas de donnée",
    "pas assez", "insuffisant", "ne dispose pas", "je n'ai pas", "n'ai aucune",
    "no history", "no historical", "no data", "not enough", "don't have", "do not have",
]  # fmt: skip
UNAVAILABLE_WORDS = [
    "indisponible", "pas disponible", "n'est pas disponible", "ne sont pas disponibles",
    "pas encore de prévision", "pas de prévision", "impossible", "pas accessible",
    "unavailable", "not available", "no forecast", "can't access", "cannot access",
]  # fmt: skip
RAIN_WORDS = ["pluie", "pleut", "pleuvoir", "averse", "rain", "shower"]

EN_WORDS = {"the", "and", "you", "is", "are", "to", "of", "with", "for", "your", "it", "at"}
FR_WORDS = {"le", "la", "les", "et", "vous", "tu", "est", "de", "des", "pour", "avec", "une"}


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def claimed_waits(answer: str) -> list[float]:
    """Every duration in minutes written in the answer."""
    values = []
    for low, high in WAIT_RE.findall(answer):
        values.append(_num(low))
        if high:
            values.append(_num(high))
    return values


def numbers_in(outputs: list[str]) -> set[float]:
    return {float(n) for text in outputs for n in NUMBER_RE.findall(text)}


def days_values(outputs: list[str]) -> set[int]:
    """days_observed / coverage.days found in the tool outputs."""
    found: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"days_observed", "days"} and isinstance(value, int):
                    found.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for text in outputs:
        try:
            walk(json.loads(text))
        except ValueError:
            continue
    return found


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    fr = sum(w in FR_WORDS for w in words)
    return "en" if en > fr else "fr"


def _contains_any(text: str, needles: list[str]) -> str | None:
    lowered = text.lower()
    return next((n for n in needles if n.lower() in lowered), None)


# --------------------------------------------------------------------------- checks


def check_mode(t: Transcript, expect: dict) -> list[Check]:
    checks = []
    if "mode" in expect:
        ok = t.mode == expect["mode"]
        checks.append(Check("mode", "mode", ok, f"got {t.mode}, expected {expect['mode']}"))
    if "date" in expect:
        ok = t.target_date == expect["date"]
        checks.append(
            Check("mode", "target date", ok, f"got {t.target_date}, expected {expect['date']}")
        )
    return checks


def check_tools(t: Transcript, expect: dict) -> list[Check]:
    checks = []
    called = t.tools_called
    for tool in expect.get("tools", []):
        checks.append(
            Check("tool_choice", f"calls {tool}", tool in called, f"called {sorted(called)}")
        )
    for tool in expect.get("forbid_tools", []):
        checks.append(Check("tool_choice", f"never calls {tool}", tool not in called, ""))
    if any_tools := expect.get("any_tools"):
        ok = bool(called & set(any_tools))
        checks.append(
            Check("tool_choice", f"calls one of {any_tools}", ok, f"called {sorted(called)}")
        )
    for tool, expected_args in expect.get("args", {}).items():
        calls = [args for name, args in t.tool_calls if name == tool]
        ok = any(
            all(str(args.get(k)) == str(v) for k, v in expected_args.items()) for args in calls
        )
        checks.append(Check("tool_choice", f"{tool} args {expected_args}", ok, f"calls: {calls}"))
    return checks


def _close(claimed: float, source: float) -> bool:
    """Rounding is fine ("≈ 20 min" for an average of 22.5), inventing is not."""
    return abs(claimed - source) <= max(2.5, 0.1 * source)


def check_no_invented_waits(t: Transcript, question: str) -> Check:
    """Every "N min" of the answer must exist (after rounding) in a tool output or in the
    question itself."""
    allowed = numbers_in(t.tool_outputs) | numbers_in([question])
    invented = [w for w in claimed_waits(t.answer) if not any(_close(w, a) for a in allowed)]
    return Check("no_hallucination", "no invented wait time", not invented, f"invented: {invented}")


def check_content(t: Transcript, checks_spec: dict) -> list[Check]:
    checks = []
    answer = t.answer
    if checks_spec.get("mentions_days_of_data"):
        days = days_values(t.tool_outputs)
        written = {int(d) for d in DAYS_RE.findall(answer)}
        ok = bool(days & written)
        checks.append(
            Check(
                "transparency", "says how many days of data", ok, f"tools {days}, answer {written}"
            )
        )
    if checks_spec.get("mentions_no_history"):
        hit = _contains_any(answer, NO_HISTORY_WORDS)
        checks.append(Check("transparency", "says data is missing", bool(hit), f"matched {hit!r}"))
    if checks_spec.get("mentions_unavailable"):
        hit = _contains_any(answer, UNAVAILABLE_WORDS)
        checks.append(
            Check("transparency", "says data is unavailable", bool(hit), f"matched {hit!r}")
        )
    if checks_spec.get("mentions_rain"):
        hit = _contains_any(answer, RAIN_WORDS)
        checks.append(Check("content", "takes the rain into account", bool(hit), ""))
    if needles := checks_spec.get("must_include_any"):
        hit = _contains_any(answer, needles)
        checks.append(Check("content", f"mentions one of {needles}", bool(hit), f"matched {hit!r}"))
    for banned in checks_spec.get("must_not_include", []):
        ok = banned.lower() not in answer.lower()
        checks.append(Check("no_hallucination", f"does not mention {banned!r}", ok, ""))
    for pattern in checks_spec.get("must_not_match", []):
        match = re.search(pattern, answer, re.IGNORECASE)
        detail = f"found {match.group(0)!r}" if match else ""
        checks.append(Check("no_hallucination", f"no match for {pattern!r}", not match, detail))
    if lang := checks_spec.get("language"):
        got = detect_language(answer)
        checks.append(Check("language", f"answers in {lang}", got == lang, f"detected {got}"))
    return checks


def run_checks(t: Transcript, case: dict) -> list[Check]:
    expect, spec = case.get("expect", {}), case.get("checks", {})
    return [
        *check_mode(t, expect),
        *check_tools(t, expect),
        check_no_invented_waits(t, case["question"]),
        *check_content(t, spec),
    ]
