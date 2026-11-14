"""Match provider events to NBA game ids and provider player names to internal player ids.
Ambiguous or missing matches are kept as UNMATCHED and are never shown as model-comparable lines."""
from __future__ import annotations
import re
from datetime import datetime

from ..core.names import name_key_from_full, strip_accents


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_accents(s or "").lower())


def team_index(teams: list[dict]) -> dict[int, set[str]]:
    """team_id -> accepted normalised names (full name, nickname, tricode)."""
    out = {}
    for t in teams:
        nick = t.get("name") or ""
        city = t.get("city") or ""
        out[t["team_id"]] = {_norm(f"{city} {nick}"), _norm(nick), _norm(t.get("tricode") or "")}
    return out


def _team_of(name: str, idx: dict[int, set[str]]) -> int | None:
    n = _norm(name)
    if not n:
        return None
    hits = [tid for tid, keys in idx.items() if n in keys or any(k and len(k) >= 4 and n.endswith(k) for k in keys)]
    if not hits:   # SGO teamID style 'LOS_ANGELES_LAKERS_NBA'
        n2 = _norm(re.sub(r"_NBA$", "", name or ""))
        hits = [tid for tid, keys in idx.items() if any(k and len(k) >= 4 and n2.endswith(k) for k in keys)]
    return hits[0] if len(set(hits)) == 1 else None


def match_event(home: str, away: str, commence: str, games: list[dict], idx) -> tuple[str | None, str]:
    h, a = _team_of(home, idx), _team_of(away, idx)
    if h is None or a is None:
        return None, "UNMATCHED_TEAM"
    try:
        t0 = datetime.fromisoformat((commence or "").replace("Z", "+00:00"))
    except ValueError:
        t0 = None
    cand = [g for g in games if g["home_team_id"] == h and g["away_team_id"] == a]
    if t0 is not None:
        cand = [g for g in cand if abs((datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00")) - t0).total_seconds()) <= 8 * 3600]
    if len(cand) == 1:
        return cand[0]["game_id"], "MATCHED"
    return None, "UNMATCHED_GAME" if not cand else "AMBIGUOUS_GAME"


def player_key_from_provider(name: str | None, provider_player_id: str | None) -> str | None:
    if name:
        return name_key_from_full(name)
    if provider_player_id:    # 'ANTHONY_EDWARDS_1_NBA' -> 'Anthony Edwards'
        base = re.sub(r"_\d+_NBA$", "", provider_player_id)
        return name_key_from_full(base.replace("_", " ").title())
    return None


def match_player(pkey: str | None, roster: dict[str, str]) -> tuple[str | None, str]:
    """roster: name_key -> player_id for the two teams of the matched game (snapshot roster)."""
    if not pkey:
        return None, "UNMATCHED_PLAYER"
    if pkey in roster:
        return roster[pkey], "MATCHED"
    last, first = pkey.split("|", 1)
    def compatible(f2: str) -> bool:      # nickname / short form: one first name is a >=3-letter prefix of the other
        a, b = sorted((first, f2), key=len)
        return len(a) >= 3 and b.startswith(a)
    cand = [pid for k, pid in roster.items() if k.split("|")[0] == last and compatible(k.split("|")[1])]
    if len(cand) == 1:
        return cand[0], "MATCHED_FUZZY"
    return None, "UNMATCHED_PLAYER" if not cand else "AMBIGUOUS_PLAYER"
