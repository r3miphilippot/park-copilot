"""The eval harness itself: checks, fixtures and golden set consistency (no LLM call)."""

from pathlib import Path

import pytest
import yaml

from app.tools import TOOLS, ToolError
from app.tools.history import HourlyProfile
from app.tools.rides import CATALOG, RIDES_BY_ID

SINGLE_RIDER_IDS = {r.single_rider_id for r in CATALOG if r.single_rider_id}
from evals.checks import (
    Transcript,
    check_content,
    check_known_rides,
    check_no_invented_waits,
    check_tools,
    claimed_waits,
    days_values,
    detect_language,
)
from evals.run_evals import FIXTURES_DIR, fixture_tool, load_fixture

GOLDEN = yaml.safe_load((Path("evals") / "golden_set.yaml").read_text(encoding="utf-8"))
TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}


def transcript(answer: str, outputs: list[str] | None = None, calls=None) -> Transcript:
    return Transcript(answer=answer, mode=None, target_date=None,
                      tool_calls=calls or [], tool_outputs=outputs or [])  # fmt: skip


# --------------------------------------------------------------------------- checks


def test_claimed_waits_formats():
    text = "Pirates : 25 min, Big Thunder 20-30 minutes, Dumbo ≈ 22,5 min, Crush 15 à 20 min."
    assert claimed_waits(text) == [25, 20, 30, 22.5, 15, 20]
    assert claimed_waits("09:30 – Big Thunder Mountain") == []  # a time is not a wait


def test_invented_wait_is_caught():
    outputs = ['{"ride": "Pirates", "avg_wait": 22.5, "days_observed": 6}']
    assert check_no_invented_waits(transcript("≈ 23 min (6 jours)", outputs), "q").passed
    assert check_no_invented_waits(transcript("≈ 20 min", outputs), "q").passed  # ±2 rounding
    failed = check_no_invented_waits(transcript("Peter Pan : 45 min", outputs), "q")
    assert not failed.passed and "45" in failed.detail


def test_days_of_data_must_match_the_tools():
    outputs = ['{"coverage": {"days": 42}, "rides": [{"days_observed": 6}]}']
    assert days_values(outputs) == {42, 6}
    ok = check_content(
        transcript("≈ 20 min, sur 6 jours de données", outputs), {"mentions_days_of_data": True}
    )
    ko = check_content(
        transcript("≈ 20 min, sur 9 jours", outputs), {"mentions_days_of_data": True}
    )
    assert ok[0].passed and not ko[0].passed


def test_language_detection():
    assert detect_language("You should arrive early and ride Big Thunder Mountain first.") == "en"
    assert (
        detect_language("Tu devrais arriver tôt et faire Big Thunder Mountain en premier.") == "fr"
    )


def test_tool_expectations():
    t = transcript("", calls=[("get_weather", {"date": "2026-07-18"}), ("search_park_guide", {})])
    checks = check_tools(t, {
        "tools": ["get_weather"], "forbid_tools": ["get_live_wait_times"],
        "args": {"get_weather": {"date": "2026-07-18"}}, "any_tools": ["get_typical_wait"],
    })  # fmt: skip
    assert [c.passed for c in checks] == [True, True, False, True]


def test_full_day_plan_checks():
    plan = "\n".join(
        [f"- {h:02d}:30 – attraction {h}" for h in range(9, 21)] + ["21:30 – spectacle nocturne"]
    )
    short = "09:30 – Dumbo\n10:15 – Pirates\n**11:00** – temps calme\n18:00 – sortie du parc"
    spec = {"min_timed_steps": 10, "mentions_evening_show": True}
    assert [c.passed for c in check_content(transcript(plan), spec)] == [True, True]
    assert [c.passed for c in check_content(transcript(short), spec)] == [False, False]


def test_known_rides_only():
    # Lines from a real answer that mixed in rides of other Disney parks.
    tester_answer = (
        "09:30 – Space Mountain (Disneyland Park)\n"
        "10:00 – Indiana Jones et la dernière croisade (Adventure World)\n"
        "13:30 – Jungle Cruise (Adventure World)\n"
        "11:30 – Lunch rapide"
    )
    good_plan = (
        "08:30 – Arrivée et contrôles de sécurité\n"
        "09:12 – Star Wars Hyperspace Mountain (Single Rider, ≈ 5 min)\n"
        "09:40 – Indiana Jones™ and the Temple of Peril (≈ 10 min)\n"
        "11:36 – Déjeuner\n"
        "**20:40** – Big Thunder Mountain (≈ 58 min)\n"
        "22:00 – Spectacle nocturne (horaire à vérifier)"
    )
    bad = check_known_rides(tester_answer)
    assert not bad.passed
    assert "jungle cruise" in bad.detail and "space mountain" in bad.detail
    assert check_known_rides(good_plan).passed  # "Hyperspace Mountain" is not "Space Mountain"


def test_forbidden_mentions_and_patterns():
    spec = {"must_not_include": ["FastPass"], "must_not_match": [r"\d+\s*cm"]}
    checks = check_content(transcript("Prends un FastPass, taille minimale 102 cm"), spec)
    assert [c.passed for c in checks] == [False, False]


# --------------------------------------------------------------------------- fixtures & golden set


@pytest.mark.parametrize("path", sorted(FIXTURES_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_fixture_matches_its_tool_model(path):
    kind = path.stem.split("_")[0]
    if kind == "profile":  # history read by plan_day, not a tool output
        profile = HourlyProfile.model_validate_json(path.read_text(encoding="utf-8"))
        assert profile.waits and all(
            r in RIDES_BY_ID or r in SINGLE_RIDER_IDS for r in profile.waits
        )
        return
    prefix = {"typical": "get_typical_wait", "weather": "get_weather",
              "compare": "compare_live_vs_typical", "live": "get_live_wait_times"}  # fmt: skip
    tool = TOOLS_BY_NAME[prefix[kind]]
    assert not isinstance(load_fixture(tool, path.stem), ToolError)


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["id"])
def test_golden_case_is_consistent(case):
    assert {"id", "question", "now", "expect"} <= set(case)
    expect = case["expect"]
    for name in [*expect.get("tools", []), *expect.get("forbid_tools", []),
                 *expect.get("any_tools", []), *expect.get("args", {})]:  # fmt: skip
        assert name in TOOLS_BY_NAME, name
    for ref in case.get("fixtures", {}).values():
        refs = ref.values() if isinstance(ref, dict) else [ref]
        for r in refs:
            assert r.startswith("error:") or (FIXTURES_DIR / f"{r}.json").exists(), r


def test_golden_set_size_and_unique_ids():
    ids = [c["id"] for c in GOLDEN]
    assert 15 <= len(ids) <= 20 and len(ids) == len(set(ids))


def test_fixture_tool_keeps_schema_and_filters_rides():
    real = TOOLS_BY_NAME["get_typical_wait"]
    fake = fixture_tool(real, {"get_typical_wait": "typical_saturday_disneyland_park"})
    assert fake.__name__ == real.__name__ and fake.__doc__ == real.__doc__
    only_pirates = fake(ride="pirates", weekday=5)
    assert [r.ride for r in only_pirates.rides] == ["Pirates of the Caribbean"]
    nothing = fake(ride="nonexistent")
    assert nothing.rides == [] and "No observation" in nothing.note


def test_fixture_tool_simulates_errors():
    real = TOOLS_BY_NAME["get_weather"]
    fake = fixture_tool(real, {"get_weather": "error:API down"})
    result = fake(date="2026-07-18")
    assert isinstance(result, ToolError) and result.error == "API down"
