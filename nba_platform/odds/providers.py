"""Odds providers. Each provider turns a real API response into normalised Quotes.

Keys come ONLY from environment variables (GitHub repository secrets). Without a key a provider
reports NO_KEY and nothing is fetched. Nothing here fabricates prices: a quote is produced only from
a price present in the provider's response."""
from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .markets import SGO_PLAYER_STATS, TOA_GAME_MARKETS, TOA_PLAYER_MARKETS
from .model import Quote


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace("+", ""))
    except ValueError:
        return None


def _price(x):
    v = _num(x)
    if v is None or (-100 < v < 100):
        return None
    return int(round(v))


@dataclass
class FetchResult:
    provider: str
    endpoint: str
    requested_at: str
    http_status: int = 0
    ok: bool = False
    error: str | None = None
    cost_units: int = 0
    n_events: int = 0
    quotes: list = field(default_factory=list)
    raw_sha256: str | None = None
    headers: dict = field(default_factory=dict)
    events: list = field(default_factory=list)    # [(provider_event_id, home, away, commence)]


class Provider:
    name = "base"
    env_key = ""
    unit = "requests"
    monthly_limit = 0

    def __init__(self, key: str | None = None, session=None):
        self.key = key if key is not None else os.environ.get(self.env_key, "").strip()
        self.session = session

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _get(self, url, params=None, headers=None, timeout=30):
        import requests
        s = self.session or requests
        return s.get(url, params=params or {}, headers=headers or {}, timeout=timeout)


class SportsGameOdds(Provider):
    """https://api.sportsgameodds.com/v2/events — billed per event object returned."""
    name = "sportsgameodds"
    env_key = "SGO_API_KEY"
    unit = "objects"
    monthly_limit = 2500            # free Amateur plan
    BASE = "https://api.sportsgameodds.com/v2/events"

    def fetch(self, starts_after: str, starts_before: str, max_pages: int = 5) -> FetchResult:
        now = datetime.now(timezone.utc).isoformat()
        res = FetchResult(self.name, "v2/events", now)
        if not self.configured:
            res.error = "NO_KEY"
            return res
        params = {"leagueID": "NBA", "oddsAvailable": "true", "startsAfter": starts_after,
                  "startsBefore": starts_before, "limit": 50}
        events, h = [], hashlib.sha256()
        for _ in range(max_pages):
            try:
                r = self._get(self.BASE, params, {"x-api-key": self.key})
            except Exception as e:                      # network failure: never raise into the cycle
                res.error = f"NETWORK:{type(e).__name__}"
                return res
            res.http_status = r.status_code
            if r.status_code != 200:
                res.error = f"HTTP_{r.status_code}"
                return res
            body = r.json()
            h.update(r.content)
            evs = body.get("data") or []
            events += evs
            res.cost_units += len(evs)
            cur = body.get("nextCursor")
            if not cur or not evs:
                break
            params["cursor"] = cur
        res.raw_sha256 = h.hexdigest()
        res.ok = True
        res.n_events = len(events)
        for ev in events:
            res.quotes += self.parse_event(ev, now)
            t = ev.get("teams") or {}
            res.events.append((ev.get("eventID"), _team_name(t.get("home")), _team_name(t.get("away")),
                               (ev.get("status") or {}).get("startsAt")))
        return res

    @staticmethod
    def parse_event(ev: dict, captured_at: str = "") -> list[Quote]:
        out = []
        eid = str(ev.get("eventID"))
        teams = ev.get("teams") or {}
        home, away = _team_name(teams.get("home")), _team_name(teams.get("away"))
        start = (ev.get("status") or {}).get("startsAt") or ""
        players = ev.get("players") or {}
        for odd_id, o in (ev.get("odds") or {}).items():
            stat, ent, period, bt, side = (o.get("statID"), o.get("statEntityID"), o.get("periodID"),
                                           o.get("betTypeID"), o.get("sideID"))
            if period != "game" or None in (stat, ent, bt, side):
                continue
            kind = market = team_side = pname = pid = None
            if ent in ("home", "away", "all") and stat == "points":
                kind = "GAME"
                if bt == "ml" and side in ("home", "away"):
                    market = "ML"
                elif bt == "sp" and side in ("home", "away"):
                    market = "SPREAD"
                elif bt == "ou" and side in ("over", "under"):
                    market = "TOTAL" if ent == "all" else "TEAM_TOTAL"
                    team_side = None if ent == "all" else ent
            elif ent not in ("home", "away", "all") and stat in SGO_PLAYER_STATS and bt == "ou" \
                    and side in ("over", "under"):
                kind, market, pid = "PLAYER", SGO_PLAYER_STATS[stat], ent
                pl = players.get(ent) or {}
                pname = pl.get("name") or " ".join(x for x in (pl.get("firstName"), pl.get("lastName")) if x) or None
            if market is None:
                continue
            for book, b in (o.get("byBookmaker") or {}).items():
                price = _price(b.get("odds"))
                if price is None:
                    continue
                line = None
                if market == "SPREAD":
                    line = _num(b.get("spread"))
                elif market != "ML":
                    line = _num(b.get("overUnder"))
                if market != "ML" and line is None:
                    continue
                out.append(Quote(SportsGameOdds.name, book, eid, home, away, start, kind, market, side, price, line,
                                 team_side, pname, pid, b.get("lastUpdatedAt"), bool(b.get("available", True))))
        return out


def _team_name(t):
    if not t:
        return ""
    n = t.get("names") or {}
    return n.get("long") or t.get("teamID") or ""


class TheOddsAPI(Provider):
    """https://api.the-odds-api.com/v4 — billed in credits = markets x regions per call."""
    name = "the_odds_api"
    env_key = "ODDS_API_KEY"
    unit = "credits"
    monthly_limit = 500              # free Starter plan
    BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
    GAME_MARKETS = ("h2h", "spreads", "totals")

    def fetch(self, starts_after: str, starts_before: str, prop_event_ids=(), prop_markets=()) -> FetchResult:
        now = datetime.now(timezone.utc).isoformat()
        res = FetchResult(self.name, "v4/odds", now)
        if not self.configured:
            res.error = "NO_KEY"
            return res
        base = {"apiKey": self.key, "regions": "us", "oddsFormat": "american", "dateFormat": "iso",
                "commenceTimeFrom": starts_after[:19] + "Z", "commenceTimeTo": starts_before[:19] + "Z"}
        h = hashlib.sha256()
        calls = [(f"{self.BASE}/odds", {**base, "markets": ",".join(self.GAME_MARKETS)}, len(self.GAME_MARKETS))]
        for eid in prop_event_ids:
            calls.append((f"{self.BASE}/events/{eid}/odds", {**base, "markets": ",".join(prop_markets)}, len(prop_markets)))
        for url, params, est in calls:
            try:
                r = self._get(url, params)
            except Exception as e:
                res.error = f"NETWORK:{type(e).__name__}"
                break
            res.http_status = r.status_code
            res.headers = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
            if r.status_code != 200:
                res.error = f"HTTP_{r.status_code}"
                break
            last = r.headers.get("x-requests-last")
            res.cost_units += int(last) if last and str(last).isdigit() else est
            h.update(r.content)
            body = r.json()
            for ev in body if isinstance(body, list) else [body]:
                res.quotes += self.parse_event(ev)
                res.events.append((ev.get("id"), ev.get("home_team"), ev.get("away_team"), ev.get("commence_time")))
            time.sleep(0.2)
        res.raw_sha256 = h.hexdigest()
        res.ok = res.error is None
        res.n_events = len({e[0] for e in res.events})
        return res

    @staticmethod
    def parse_event(ev: dict) -> list[Quote]:
        out = []
        eid, home, away, start = str(ev.get("id")), ev.get("home_team") or "", ev.get("away_team") or "", ev.get("commence_time") or ""
        for bk in ev.get("bookmakers") or []:
            book = bk.get("key")
            for mk in bk.get("markets") or []:
                key = mk.get("key")
                upd = mk.get("last_update") or bk.get("last_update")
                for oc in mk.get("outcomes") or []:
                    price = _price(oc.get("price"))
                    if price is None:
                        continue
                    name, desc, pt = oc.get("name"), oc.get("description"), _num(oc.get("point"))
                    if key in TOA_GAME_MARKETS:
                        market = TOA_GAME_MARKETS[key]
                        if market in ("ML", "SPREAD"):
                            side = "home" if name == home else "away" if name == away else None
                            if side is None or (market == "SPREAD" and pt is None):
                                continue
                            out.append(Quote(TheOddsAPI.name, book, eid, home, away, start, "GAME", market, side, price,
                                             pt if market == "SPREAD" else None, None, None, None, upd))
                        else:
                            side = (name or "").lower()
                            if side not in ("over", "under") or pt is None:
                                continue
                            ts = None
                            if market == "TEAM_TOTAL":
                                ts = "home" if desc == home else "away" if desc == away else None
                                if ts is None:
                                    continue
                            out.append(Quote(TheOddsAPI.name, book, eid, home, away, start, "GAME", market, side, price,
                                             pt, ts, None, None, upd))
                    elif key in TOA_PLAYER_MARKETS:
                        side = (name or "").lower()
                        if side not in ("over", "under") or pt is None or not desc:
                            continue
                        q = Quote(TheOddsAPI.name, book, eid, home, away, start, "PLAYER", TOA_PLAYER_MARKETS[key], side,
                                  price, pt, None, desc, None, upd)
                        if key == "player_field_goals":
                            q.notes.append("FGM_ASSUMED_FROM_player_field_goals")
                        out.append(q)
        return out


PROVIDERS = {p.name: p for p in (SportsGameOdds, TheOddsAPI)}
