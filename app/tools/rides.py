"""Ride catalogue of both parks + tool 6: list_rides.

A closed, reviewed list: ride ids and names come from Queue-Times (so they match the collected
history), the other attributes are curated. The agent must only cite rides from here: without
it, the LLM filled gaps from its general knowledge of other Disney parks ("Jungle Cruise" is
not in Paris).

Rides are matched by id, never by name (some Queue-Times names contain invisible characters).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.config import ParkKey
from app.tools.common import ToolError, tool_guard

Kind = Literal["ride", "show", "walkthrough", "meet", "playground", "transport"]
Thrill = Literal["high", "medium", "low"]
Popularity = Literal["headliner", "popular", "standard"]


class Ride(BaseModel):
    ride_id: int  # Queue-Times id
    name: str
    park: ParkKey
    land: str
    kind: Kind
    indoor: bool
    thrill: Thrill
    popularity: Popularity  # how fast the queue builds up; used when there is no history
    single_rider_id: int | None = None  # Queue-Times id of the Single Rider line


def _r(ride_id, name, park, land, kind, indoor, thrill="low", popularity="standard", sr=None):
    return Ride(ride_id=ride_id, name=name, park=park, land=land, kind=kind, indoor=indoor,
                thrill=thrill, popularity=popularity, single_rider_id=sr)  # fmt: skip


DLP, AWP = "disneyland_park", "adventure_world"

CATALOG: list[Ride] = [
    # Disneyland Park ------------------------------------------------------------------------
    _r(2, "Indiana Jones™ and the Temple of Peril", DLP, "Adventureland", "ride", False, "high", "popular", 7306),
    _r(3, "Pirates of the Caribbean", DLP, "Adventureland", "ride", True, "low", "popular"),
    _r(1, "La Cabane des Robinson", DLP, "Adventureland", "walkthrough", False),
    _r(2702, "Adventure Isle", DLP, "Adventureland", "playground", False),
    _r(2703, "Le Passage Enchanté d'Aladdin", DLP, "Adventureland", "walkthrough", True),
    _r(2704, "Pirates' Beach", DLP, "Adventureland", "playground", False),
    _r(2705, "Pirate Galleon", DLP, "Adventureland", "walkthrough", False),
    _r(8, "Star Wars Hyperspace Mountain", DLP, "Discoveryland", "ride", True, "high", "headliner", 7278),
    _r(9, "Star Tours: The Adventures Continue", DLP, "Discoveryland", "ride", True, "medium", "popular"),
    _r(5, "Buzz Lightyear Laser Blast", DLP, "Discoveryland", "ride", True, "low", "popular"),
    _r(4, "Autopia", DLP, "Discoveryland", "ride", False),
    _r(7, "Orbitron", DLP, "Discoveryland", "ride", False),
    _r(6, "Les Mystères du Nautilus", DLP, "Discoveryland", "walkthrough", True),
    _r(2707, "Mickey's PhilharMagic", DLP, "Discoveryland", "show", True),
    _r(4573, "Welcome to Starport: A Star Wars Encounter", DLP, "Discoveryland", "meet", True),
    _r(22, "Peter Pan's Flight", DLP, "Fantasyland", "ride", True, "low", "headliner"),
    _r(19, '"it\'s a small world"', DLP, "Fantasyland", "ride", True, "low", "popular"),
    _r(18, "Dumbo the Flying Elephant", DLP, "Fantasyland", "ride", False, "low", "popular"),
    _r(15, "Blanche-Neige et les Sept Nains", DLP, "Fantasyland", "ride", True),
    _r(23, "Les Voyages de Pinocchio", DLP, "Fantasyland", "ride", True),
    _r(17, "Casey Jr. – le Petit Train du Cirque", DLP, "Fantasyland", "ride", False),
    _r(21, "Le Pays des Contes de Fées", DLP, "Fantasyland", "ride", False),
    _r(16, "Le Carrousel de Lancelot", DLP, "Fantasyland", "ride", False),
    _r(20, "Mad Hatter's Tea Cups", DLP, "Fantasyland", "ride", False),
    _r(14, "Alice's Curious Labyrinth", DLP, "Fantasyland", "walkthrough", False),
    _r(2710, "La Tanière du Dragon", DLP, "Fantasyland", "walkthrough", True),
    _r(13, "Meet Mickey Mouse", DLP, "Fantasyland", "meet", True),
    _r(24, "Princess Pavilion", DLP, "Fantasyland", "meet", True),
    _r(25, "Big Thunder Mountain", DLP, "Frontierland", "ride", False, "high", "headliner"),
    _r(26, "Phantom Manor", DLP, "Frontierland", "ride", True, "low", "popular"),
    _r(28, "Thunder Mesa Riverboat Landing", DLP, "Frontierland", "ride", False),
    _r(27, "River Rogue Keelboats", DLP, "Frontierland", "ride", False),
    _r(2713, "Frontierland Playground", DLP, "Frontierland", "playground", False),
    _r(12, "Main Street Vehicles", DLP, "Main Street U.S.A", "transport", False),
    _r(2708, "Disneyland Railroad", DLP, "Main Street U.S.A", "transport", False),
    # Disney Adventure World -----------------------------------------------------------------
    _r(10848, "Avengers Assemble: Flight Force", AWP, "Marvel Avengers Campus", "ride", True, "high", "headliner", 10849),
    _r(10845, "Spider-Man W.E.B. Adventure", AWP, "Marvel Avengers Campus", "ride", True, "low", "popular", 10846),
    _r(40, "The Twilight Zone Tower of Terror", AWP, "Production Courtyard", "ride", True, "high", "headliner"),
    _r(32, "Crush's Coaster", AWP, "Toon Studio", "ride", True, "high", "headliner", 7277),
    _r(37, "Ratatouille : L'Aventure Totalement Toquée de Rémy", AWP, "Toon Studio", "ride", True, "low", "headliner", 7279),
    _r(34, "RC Racer", AWP, "Toon Studio", "ride", False, "high", "popular", 7280),
    _r(35, "Toy Soldiers Parachute Drop", AWP, "Toon Studio", "ride", False, "medium", "popular", 7281),
    _r(36, "Slinky Dog Zigzag Spin", AWP, "Toon Studio", "ride", False),
    _r(29, "Cars ROAD TRIP", AWP, "Toon Studio", "ride", False),
    _r(31, "Cars Quatre Roues Rallye", AWP, "Toon Studio", "ride", False),
    _r(33, "Les Tapis Volants - Flying Carpets Over Agrabah", AWP, "Toon Studio", "ride", False),
    _r(15413, "Frozen Ever After", AWP, "World of Frozen", "ride", True, "low", "headliner"),
    _r(15415, "Raiponce Tangled Spin", AWP, "World of Frozen", "ride", False),
]  # fmt: skip

RIDES_BY_ID = {ride.ride_id: ride for ride in CATALOG}


class RideCatalog(BaseModel):
    park: ParkKey
    rides: list[Ride]
    note: str = (
        "Official list: only cite these rides, in this park. thrill = sensations level; "
        "single_rider_id set = a Single Rider line exists (shorter wait, the group rides apart)."
    )


@tool_guard
def list_rides(park: ParkKey) -> RideCatalog | ToolError:
    """List the rides, shows and attractions of one park with their land, indoor/outdoor,
    thrill level (high / medium / low) and whether a Single Rider line exists.

    This is the official list: never cite a ride that is not in it.

    Args:
        park: "disneyland_park" or "adventure_world".
    """
    return RideCatalog(park=park, rides=[r for r in CATALOG if r.park == park])
