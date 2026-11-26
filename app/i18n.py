"""Texts that the tools write for the visitor (plan step titles and notes, crowd levels and
reasons), in the language of the answer.

The LLM copies these texts as they are, and smaller models (the fallback) forget to translate
them: they must already be in the right language. The agent sets the language for each tool
call (`answer_language`); the MCP server, the tests and the CLI get English.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

Lang = Literal["fr", "en"]

# A ContextVar (like the frozen clock): concurrent requests do not see each other's language,
# and asyncio.to_thread copies it into the tool threads.
_lang: ContextVar[Lang] = ContextVar("answer_lang", default="en")

TEXTS: dict[str, dict[Lang, str]] = {
    # plan_day steps
    "arrival": {"en": "Arrival and security checks", "fr": "Arrivée et contrôles de sécurité"},
    "opens_at": {"en": "Park opens at {time}.", "fr": "Ouverture du parc à {time}."},
    "rides_start": {
        "en": "Rides usually start around {time}.",
        "fr": "Les attractions ouvrent en général vers {time}.",
    },
    "emh_from": {
        "en": "Extra Magic Hours from {start}, park opens at {opening}.",
        "fr": "Extra Magic Hours dès {start}, ouverture du parc à {opening}.",
    },
    "lunch": {"en": "Lunch", "fr": "Déjeuner"},
    "lunch_note": {
        "en": "Quick-service, before the 12:00-14:00 peak.",
        "fr": "Restauration rapide, avant le pic de 12:00-14:00.",
    },
    "dinner": {"en": "Dinner", "fr": "Dîner"},
    "dinner_note": {
        "en": "Early dinner, to avoid the evening peak.",
        "fr": "Dîner tôt, pour éviter le pic du soir.",
    },
    "night_show": {"en": "Night show at the castle", "fr": "Spectacle nocturne au château"},
    "night_show_named": {"en": "Night show: {name}", "fr": "Spectacle nocturne : {name}"},
    "night_show_note": {
        "en": "Official time today; find a spot about 20 min before.",
        "fr": "Heure officielle du jour ; se placer environ 20 min avant.",
    },
    "night_show_estimated": {
        "en": "Usually shortly before closing ({closing}): exact time published on the day "
        "in the official app; find a spot early.",
        "fr": "En général peu avant la fermeture ({closing}) : heure exacte publiée le jour "
        "même dans l'application officielle ; se placer en avance.",
    },
    "parade": {"en": "Parade: {name}", "fr": "Parade : {name}"},
    "parade_note": {
        "en": "Official time today; find a spot about 15 min before. Queues elsewhere drop "
        "during the parade.",
        "fr": "Heure officielle du jour ; se placer environ 15 min avant. Les files baissent "
        "ailleurs pendant la parade.",
    },
    "second_ride": {"en": "second ride", "fr": "deuxième tour"},
    "live_wait": {"en": "live wait", "fr": "attente en direct"},
    # crowd calendar
    "low": {"en": "quiet", "fr": "calme"},
    "moderate": {"en": "moderate", "fr": "modérée"},
    "busy": {"en": "busy", "fr": "chargée"},
    "very_busy": {"en": "very busy", "fr": "très chargée"},
    "school_holidays": {
        "en": "French school holidays ({name}), zones {zones}",
        "fr": "vacances scolaires ({name}), zones {zones}",
    },
    "public_holiday": {"en": "French public holiday ({name})", "fr": "jour férié ({name})"},
}

WEEKDAY_NAMES: dict[Lang, list[str]] = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
}


def current_lang() -> Lang:
    return _lang.get()


def text(key: str, **values: str) -> str:
    return TEXTS[key][current_lang()].format(**values)


def weekday_name(weekday: int) -> str:
    return WEEKDAY_NAMES[current_lang()][weekday]


@contextmanager
def answer_language(lang: Lang) -> Iterator[None]:
    token = _lang.set(lang)
    try:
        yield
    finally:
        _lang.reset(token)
