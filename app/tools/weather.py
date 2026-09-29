"""Tool 4: hourly weather forecast for the resort, from Open-Meteo (free, no API key)."""

from __future__ import annotations

import datetime as dt
from datetime import date, timedelta

import httpx
from pydantic import BaseModel

from app.clock import now_paris
from app.tools.common import ToolError, ToolFailure, TTLCache, tool_guard

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 48.8722, 2.7758  # Marne-la-Vallée (resort)
MAX_FORECAST_DAYS = 15  # Open-Meteo accepts today + 15 days
PARK_HOURS = range(8, 24)  # hours returned to the LLM (keeps the payload short)

# An hour counts as rainy above these thresholds.
RAIN_MM = 0.2
RAIN_PROBABILITY = 50

_cache = TTLCache(ttl_s=60 * 60)

# WMO weather codes (https://open-meteo.com/en/docs), grouped.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}  # fmt: skip


class WeatherHour(BaseModel):
    time: str  # "HH:00", Paris time
    temperature_c: float | None
    feels_like_c: float | None
    precipitation_mm: float | None
    precipitation_probability: int | None  # %
    wind_kmh: float | None
    conditions: str


class WeatherForecast(BaseModel):
    date: date
    location: str = "Marne-la-Vallée"
    min_temp_c: float | None
    max_temp_c: float | None
    total_precipitation_mm: float
    rainy_hours: list[str]  # hours (08:00-23:00) where rain is likely
    rain_expected: bool
    hours: list[WeatherHour]  # 08:00 to 23:00
    source: str = "Open-Meteo.com"


@tool_guard
def get_weather(date: dt.date) -> WeatherForecast | ToolError:
    """Get the hourly weather forecast at the resort for one day (today or up to 15 days
    ahead): temperature, rain probability and amount, wind, and the rainy hours.

    Use it to plan indoor rides during rainy hours.

    Args:
        date: the day, in YYYY-MM-DD format.
    """
    # The parameter is named "date" because that is what LLMs spontaneously send: with
    # "day", gpt-oss sometimes called get_weather(date=...) and Groq rejected the call.
    day = date
    today = now_paris().date()
    if day < today:
        raise ToolFailure(f"{day} is in the past: forecasts only exist from today ({today}).")
    if day > today + timedelta(days=MAX_FORECAST_DAYS):
        raise ToolFailure(
            f"No forecast yet for {day}: available up to "
            f"{today + timedelta(days=MAX_FORECAST_DAYS)}. Plan without weather."
        )

    cached = _cache.get(day)
    if cached is not None:
        return cached

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,"
        "precipitation,weather_code,wind_speed_10m",
        "timezone": "Europe/Paris",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    response = httpx.get(OPEN_METEO_URL, params=params, timeout=10.0)
    response.raise_for_status()
    result = parse_forecast(day, response.json()["hourly"])
    _cache.set(day, result)
    return result


def parse_forecast(day: date, hourly: dict[str, list]) -> WeatherForecast:
    """Turn Open-Meteo's column-oriented `hourly` block into one row per park hour."""
    hours: list[WeatherHour] = []
    for i, stamp in enumerate(hourly["time"]):  # "2026-09-29T14:00", already Paris time
        hour = int(stamp[11:13])
        if hour not in PARK_HOURS:
            continue

        def col(name: str, i: int = i):
            values = hourly.get(name) or []
            return values[i] if i < len(values) else None

        code = col("weather_code")
        hours.append(
            WeatherHour(
                time=f"{hour:02d}:00",
                temperature_c=col("temperature_2m"),
                feels_like_c=col("apparent_temperature"),
                precipitation_mm=col("precipitation"),
                precipitation_probability=col("precipitation_probability"),
                wind_kmh=col("wind_speed_10m"),
                conditions=_WMO.get(code, "unknown") if code is not None else "unknown",
            )
        )

    rainy = [
        h.time
        for h in hours
        if (h.precipitation_mm or 0) >= RAIN_MM
        or (h.precipitation_probability or 0) >= RAIN_PROBABILITY
    ]
    temps = [h.temperature_c for h in hours if h.temperature_c is not None]
    return WeatherForecast(
        date=day,
        min_temp_c=min(temps) if temps else None,
        max_temp_c=max(temps) if temps else None,
        total_precipitation_mm=round(sum(h.precipitation_mm or 0 for h in hours), 1),
        rainy_hours=rainy,
        rain_expected=bool(rainy),
        hours=hours,
    )
