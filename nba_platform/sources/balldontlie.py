"""balldontlie.io adapter (licensed, cloud-accessible, $39.99/mo GOAT tier for NBA).

Used for: active players (with team), injuries (licensed feed, NOT official), lineups,
historical box scores. Their player ids differ from NBA personIds -> resolved via xref
(see roster.sync.IdResolver).

Field names follow the public docs at https://docs.balldontlie.io. Parsers are defensive and
raise SchemaDrift when required fields disappear. The lineup endpoint contract must be verified
on the first live call (marked VERIFY below) — no live call was possible from the build sandbox.
"""
from __future__ import annotations

import json
from typing import Iterator

from ..core.enums import AvailabilityStatus
from .http import Fetcher
from .nba_cdn import SchemaDrift

BASE = "https://api.balldontlie.io/v1"

# balldontlie injury statuses -> ours. 'Day-To-Day' is a GTD-style designation.
BDL_STATUS = {
    "out": AvailabilityStatus.OUT, "questionable": AvailabilityStatus.QUESTIONABLE,
    "doubtful": AvailabilityStatus.DOUBTFUL, "probable": AvailabilityStatus.PROBABLE,
    "day-to-day": AvailabilityStatus.GTD, "available": AvailabilityStatus.AVAILABLE,
    "suspension": AvailabilityStatus.SUSPENDED, "suspended": AvailabilityStatus.SUSPENDED,
}


class BallDontLie:
    def __init__(self, api_key: str, fetcher: Fetcher):
        self.key, self.f = api_key, fetcher

    def _paged(self, path: str, params: dict) -> Iterator[dict]:
        cursor = None
        while True:
            p = {**params, "per_page": 100}
            if cursor:
                p["cursor"] = cursor
            res = self.f.get(f"{BASE}{path}", "balldontlie", headers={"Authorization": self.key}, params=p)
            doc = json.loads(res.body)
            if "data" not in doc:
                raise SchemaDrift(f"{path}: no 'data'")
            yield from doc["data"]
            cursor = (doc.get("meta") or {}).get("next_cursor")
            if not cursor:
                return

    def active_players(self) -> list[dict]:
        return [parse_player(p) for p in self._paged("/players/active", {})]

    def injuries(self) -> list[dict]:
        return [parse_injury(i) for i in self._paged("/player_injuries", {})]

    def lineups(self, game_ids: list[int]) -> list[dict]:  # VERIFY contract on first live call
        return list(self._paged("/lineups", {"game_ids[]": game_ids}))


def parse_player(p: dict) -> dict:
    try:
        team = p.get("team") or {}
        return {"bdl_id": str(p["id"]), "first_name": p["first_name"], "last_name": p["last_name"],
                "position": p.get("position") or None, "team_abbr": team.get("abbreviation"),
                "jersey": p.get("jersey_number")}
    except KeyError as e:
        raise SchemaDrift(f"player missing {e}") from e


def parse_injury(i: dict) -> dict:
    try:
        pl = i["player"]
        raw = (i.get("status") or "").strip()
        return {"bdl_id": str(pl["id"]), "first_name": pl["first_name"], "last_name": pl["last_name"],
                "team_bdl_id": pl.get("team_id"), "raw_status": raw,
                "status": BDL_STATUS.get(raw.lower(), AvailabilityStatus.UNKNOWN),
                "description": i.get("description"), "return_date": i.get("return_date")}
    except KeyError as e:
        raise SchemaDrift(f"injury missing {e}") from e
