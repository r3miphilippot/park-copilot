"""Run the golden set against the real agent (real LLM, frozen tool outputs).

    uv run python -m evals.run_evals                      # all cases
    uv run python -m evals.run_evals --cases plan_saturday_kids,park_closed --no-judge
    uv run python -m evals.run_evals --fail-under 0.8     # non-zero exit below 80 %

Writes evals/reports/report.md and results.json, and appends the report to the GitHub job
summary when running in Actions.

Reproducibility: the date/time is frozen per case and every tool returns a frozen fixture,
except search_park_guide (local RAG, deterministic). Only the LLM varies between runs.
The agent runs on the primary model only (no fallback), so a quota hit is waited out
instead of silently evaluating another model.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import statistics
import sys
import time
import typing
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.agent.graph import UNAVAILABLE_MESSAGES, build_graph
from app.clock import frozen_now
from app.config import PARIS_TZ, get_settings
from app.llm import LLMRouter, make_provider
from app.tools import TOOLS, ToolError, get_weather
from app.tools.history import HourlyProfile
from evals.checks import Check, Transcript, run_checks

EVALS_DIR = Path(__file__).parent
FIXTURES_DIR = EVALS_DIR / "fixtures"
LIVE_TOOLS_WITH_FIXTURES = {
    "get_live_wait_times", "get_typical_wait", "compare_live_vs_typical", "get_weather",
}  # fmt: skip
JUDGE_MODEL = "openai/gpt-oss-20b"  # another model than the agent's: no self-grading
JUDGE_PASS = 3.5


# --------------------------------------------------------------------------- fixture tools


def _success_model(fn) -> type[BaseModel]:
    hints = typing.get_type_hints(fn)
    [model] = [t for t in typing.get_args(hints["return"]) if t is not ToolError]
    return model


def load_fixture(fn, ref: str) -> BaseModel:
    if ref.startswith("error:"):
        return ToolError(tool=fn.__name__, error=ref.removeprefix("error:"))
    data = json.loads((FIXTURES_DIR / f"{ref}.json").read_text(encoding="utf-8"))
    return _success_model(fn).model_validate(data)


def fixture_tool(fn, case_fixtures: dict[str, Any]):
    """Same name, signature and docstring as the real tool (so the LLM sees the same schema),
    but returns the case's frozen output."""
    name = fn.__name__
    if name not in LIVE_TOOLS_WITH_FIXTURES:
        return fn  # search_park_guide: the real local RAG

    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> BaseModel:
        ref = case_fixtures.get(name)
        if isinstance(ref, dict):  # one fixture per park
            ref = ref.get(kwargs.get("park") or "default", ref.get("default"))
        if ref is None:
            return ToolError(tool=name, error="Data unavailable for this request.")
        result = load_fixture(fn, ref)
        if name == "get_typical_wait" and not isinstance(result, ToolError):
            result = _filter_typical(result, kwargs)
        return result

    return wrapper


def _filter_typical(result, kwargs: dict):
    """Mimic the real filters on ride name and park, like the MongoDB query would."""
    rides = result.rides
    if ride := kwargs.get("ride"):
        rides = [r for r in rides if ride.lower() in r.ride.lower()]
    if park := kwargs.get("park"):
        rides = [r for r in rides if r.park == park]
    note = result.note
    if result.rides and not rides:
        note = "No observation matches these filters: say so, and rely on the park guide."
    return result.model_copy(update={"rides": rides, "note": note})


# --------------------------------------------------------------------------- judge


class Judgement(BaseModel):
    timed_plan: int = Field(
        ge=1, le=5, description="Concrete schedule with times (5 = every step timed)"
    )
    coherence: int = Field(
        ge=1, le=5, description="Logical order, rides grouped by area, realistic pace"
    )
    grounding: int = Field(
        ge=1,
        le=5,
        description="Uses the tool data (waits, days of data, weather); nothing invented",
    )
    usefulness: int = Field(
        ge=1, le=5, description="Answers the visitor's actual constraints (kids, rain, date...)"
    )
    completeness: int = Field(
        ge=1,
        le=5,
        description="A real full day: from opening to the evening show, 10+ rides, no filler",
    )
    comment: str = Field(default="", description="One sentence justifying the scores")

    @property
    def score(self) -> float:
        return statistics.mean(
            [self.timed_plan, self.coherence, self.grounding, self.usefulness, self.completeness]
        )


JUDGE_PROMPT = """You grade the answer of a theme-park planning assistant. Score each criterion
from 1 (bad) to 5 (excellent). Be strict: an invented wait time or ride means grounding <= 2.

Visitor question:
{question}

Data returned by the assistant's tools (JSON, possibly truncated):
{tools}

Assistant answer:
{answer}"""


async def judge(router: LLMRouter, question: str, t: Transcript) -> Judgement:
    tools = "\n".join(t.tool_outputs)[:6000]
    prompt = JUDGE_PROMPT.format(question=question, tools=tools, answer=t.answer)
    result = await router.ainvoke(
        lambda m: m.with_structured_output(Judgement),
        [SystemMessage("You are a strict evaluator."), HumanMessage(prompt)],
    )
    return result.value


# --------------------------------------------------------------------------- run


def _router(model: str, max_wait_s: float) -> LLMRouter:
    settings = get_settings()
    # No fallback: evaluate this exact model, and wait out the quota (retry-after) instead.
    return LLMRouter(make_provider("groq", model, settings), None, max_wait_s=max_wait_s)


class FixtureHistoryStore:
    """History for plan_day: the case's frozen hourly profiles (one fixture per park)."""

    def __init__(self, refs: dict[str, str] | None) -> None:
        self.refs = refs or {}

    def hourly_profile(self, *, park: str, weekday: int | None) -> HourlyProfile:
        ref = self.refs.get(park)
        if ref is None:
            return HourlyProfile(park=park, basis="none", days=0, waits={})
        data = json.loads((FIXTURES_DIR / f"{ref}.json").read_text(encoding="utf-8"))
        return HourlyProfile.model_validate(data)


@contextmanager
def planner_on_fixtures(case_fixtures: dict[str, Any]):
    """plan_day runs for real (it is what we evaluate) but reads frozen history and weather."""
    weather = fixture_tool(get_weather, case_fixtures)
    store = FixtureHistoryStore(case_fixtures.get("hourly_profile"))
    with (
        patch("app.tools.planner.get_history_store", lambda: store),
        patch("app.tools.planner.get_weather", lambda date: weather(date=date)),
    ):
        yield


async def run_case(case: dict, agent_router: LLMRouter) -> Transcript:
    tools = [fixture_tool(fn, case.get("fixtures", {})) for fn in TOOLS]
    graph = build_graph(agent_router, tools=tools, checkpointer=InMemorySaver())
    now = datetime.fromisoformat(case["now"]).replace(tzinfo=PARIS_TZ)
    config = {"configurable": {"thread_id": case["id"]}, "recursion_limit": 30}
    with frozen_now(now), planner_on_fixtures(case.get("fixtures", {})):
        state = await graph.ainvoke(
            {"messages": [HumanMessage(case["question"])], "lang": case.get("lang")}, config
        )
    messages = state["messages"]
    answer = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    return Transcript(
        answer=answer if isinstance(answer, str) else str(answer),
        mode=state.get("mode"),
        target_date=state.get("target_date"),
        tool_calls=[
            (c["name"], c["args"])
            for m in messages
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ],  # fmt: skip
        tool_outputs=[m.content for m in messages if isinstance(m, ToolMessage)],
    )


async def evaluate(cases: list[dict], *, use_judge: bool, pause_s: float) -> list[dict]:
    settings = get_settings()
    agent_router = _router(settings.llm_model, max_wait_s=90)
    judge_router = _router(JUDGE_MODEL, max_wait_s=90)
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
        started = time.perf_counter()
        crash = None
        for _attempt in range(3):
            try:
                transcript = await run_case(case, agent_router)
            except Exception as exc:  # a crash fails this case, not the whole run
                crash = exc
                transcript = Transcript(answer="", mode=None, target_date=None)
                break
            if transcript.answer not in UNAVAILABLE_MESSAGES.values():
                break
            print("  LLM quota hit, waiting 65 s before retrying", flush=True)
            await asyncio.sleep(65)
        if crash is None and transcript.answer in UNAVAILABLE_MESSAGES.values():
            # The free quota, not the agent, failed: report it apart, out of the scores.
            print("  SKIPPED: LLM quota still exhausted", flush=True)
            results.append({"id": case["id"], "category": case.get("category", ""),
                             "question": case["question"], "skipped": "llm_quota",
                             "passed": None, "checks": [], "judge": None, "tools": [],
                             "mode": None, "answer": ""})  # fmt: skip
            await asyncio.sleep(pause_s)
            continue
        checks = [
            Check("robustness", "no crash", crash is None, repr(crash)[:300] if crash else "")
        ]
        checks += run_checks(transcript, case) if crash is None else []
        judgement = None
        if use_judge and case.get("judge") and crash is None:
            try:
                judgement = await judge(judge_router, case["question"], transcript)
            except Exception as exc:  # a failing judge is not the agent's fault: not judged
                print(f"  judge unavailable ({type(exc).__name__}), case not graded", flush=True)
        if judgement is not None:
            checks.append(
                Check(
                    "planning_quality",
                    "LLM judge score",
                    judgement.score >= JUDGE_PASS,
                    f"{judgement.score:.2f}/5 - {judgement.comment}",
                )  # fmt: skip
            )
        failed = [c for c in checks if not c.passed]
        print(f"  {'PASS' if not failed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)})"
              + "".join(f"\n    - {c.name}: {c.detail}" for c in failed), flush=True)  # fmt: skip
        results.append({
            "id": case["id"], "category": case.get("category", ""), "question": case["question"],
            "passed": not failed, "duration_s": round(time.perf_counter() - started, 1),
            "mode": transcript.mode, "target_date": transcript.target_date,
            "tools": [name for name, _ in transcript.tool_calls],
            "judge": judgement.model_dump() | {"score": judgement.score} if judgement else None,
            "checks": [vars(c) for c in checks], "answer": transcript.answer,
        })  # fmt: skip
        if i < len(cases):
            await asyncio.sleep(pause_s)  # Groq free tier: a few thousand tokens per minute
    return results


# --------------------------------------------------------------------------- report


def build_report(results: list[dict]) -> tuple[str, float]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for c in r["checks"]:
            by_category[c["category"]].append(c["passed"])
    all_checks = [p for values in by_category.values() for p in values]
    rate = sum(all_checks) / len(all_checks) if all_checks else 0.0
    scored = [r for r in results if not r.get("skipped")]
    cases_ok = sum(bool(r["passed"]) for r in scored)
    skipped = [r["id"] for r in results if r.get("skipped")]

    lines = [
        "# Park Copilot evals",
        "",
        f"**{cases_ok}/{len(scored)} cases fully passed · {rate:.0%} of checks passed** "
        f"· model `{get_settings().llm_model}` · judge `{JUDGE_MODEL}`",
        "",
    ]
    if skipped:
        lines += [f"Not scored (free LLM quota exhausted): {', '.join(skipped)}", ""]
    lines += [
        "| Category | Pass rate | Checks |",
        "|---|---|---|",
    ]
    for category in ["robustness", "mode", "tool_choice", "no_hallucination", "transparency",
                     "content", "language", "planning_quality"]:  # fmt: skip
        values = by_category.get(category)
        if values:
            lines.append(
                f"| {category} | {sum(values) / len(values):.0%} | {sum(values)}/{len(values)} |"
            )
    judged = [r["judge"]["score"] for r in results if r["judge"]]
    if judged:
        lines += ["", f"Average planning score (LLM judge): **{statistics.mean(judged):.2f}/5**"]

    lines += ["", "| Case | Result | Mode | Tools | Failed checks |", "|---|---|---|---|---|"]
    for r in results:
        failed = "; ".join(f"{c['name']} ({c['detail']})" for c in r["checks"] if not c["passed"])
        tools = ", ".join(r["tools"]) or "-"
        mark = "⏭️" if r.get("skipped") else ("✅" if r["passed"] else "❌")
        lines.append(f"| `{r['id']}` | {mark} | {r['mode']} | {tools} | {failed or '-'} |")
    return "\n".join(lines) + "\n", rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Park Copilot golden set")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--pause", type=float, default=30.0, help="seconds between cases")
    parser.add_argument("--fail-under", type=float, help="exit 1 if the check pass rate is lower")
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "reports")
    args = parser.parse_args(argv)

    cases = yaml.safe_load((EVALS_DIR / "golden_set.yaml").read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = asyncio.run(evaluate(cases, use_judge=not args.no_judge, pause_s=args.pause))
    report, rate = build_report(results)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report)
    print("\n" + report)
    return 1 if args.fail_under is not None and rate < args.fail_under else 0


if __name__ == "__main__":
    sys.exit(main())
