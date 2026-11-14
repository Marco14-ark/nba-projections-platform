"""Active lineup resolution (spec §6, §34, §58).

Produces, for one team in one game as of time T:
  * who is definitely unavailable (OUT/REST/SUSPENDED/NOT_WITH_TEAM) -> zero minutes, always
  * who is uncertain (QUESTIONABLE/DOUBTFUL/GTD) -> availability scenarios
  * starters: CONFIRMED if a confirmed lineup exists as of T, otherwise EXPECTED (model)
  * explicit minutes caps (only if a source reported one)
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

from ..core.enums import AvailabilityStatus as S, LineupState
from ..db.store import Store
from ..injuries.monitor import InjuryMonitor
from ..roster.sync import roster_on


@dataclass(frozen=True)
class Scenario:
    name: str
    extra_out: frozenset[str]
    prob: float | None          # None => no defensible probability; show separately


@dataclass
class TeamContext:
    game_id: str
    team_id: int
    as_of: str
    roster: list[str]
    statuses: dict[str, S]
    out: set[str]
    uncertain: dict[str, S]
    confirmed_starters: list[str] | None
    lineup_state: LineupState
    minutes_caps: dict[str, float]
    flags: list[str] = field(default_factory=list)

    def available(self, scenario: Scenario | None = None) -> list[str]:
        extra = scenario.extra_out if scenario else frozenset()
        return [p for p in self.roster if p not in self.out and p not in extra]


def resolve_team(store: Store, game_id: str, team_id: int, game_date_et: str, as_of: str) -> TeamContext:
    mon = InjuryMonitor(store)
    roster = roster_on(store, team_id, game_date_et)
    st = {p: s for p, s in mon.state(game_id, as_of).items() if p in roster}
    out = {p for p, s in st.items() if s.is_definitely_out}
    uncertain = {p: s for p, s in st.items() if s.is_uncertain}
    lu = store.lineup_asof(game_id, team_id, as_of)
    flags: list[str] = []
    confirmed = None
    state = LineupState.EXPECTED
    if lu is not None and lu["state"] == "CONFIRMED":
        confirmed = json.loads(lu["starters"])
        state = LineupState.CONFIRMED
        bad = [p for p in confirmed if p in out]
        if bad:
            flags.append(f"CONFIRMED_STARTER_NOW_OUT:{','.join(bad)}")
            confirmed = [p for p in confirmed if p not in out]
            state = LineupState.EXPECTED
        # a confirmed starter is playing: uncertainty resolved
        for p in confirmed:
            uncertain.pop(p, None)
    if not roster:
        flags.append("EMPTY_ROSTER")
    return TeamContext(game_id, team_id, as_of, roster, st, out, uncertain, confirmed, state,
                       mon.minutes_caps(game_id, as_of), flags)


def expected_starters(available: list[str], start_scores: dict[str, float],
                      confirmed: list[str] | None = None) -> list[str]:
    if confirmed and len(confirmed) == 5 and all(p in available for p in confirmed):
        return list(confirmed)
    keep = [p for p in (confirmed or []) if p in available]
    rest = sorted((p for p in available if p not in keep), key=lambda p: -start_scores.get(p, 0.0))
    return (keep + rest)[:5]


def build_scenarios(ctx: TeamContext, baseline_minutes: dict[str, float],
                    play_rates: dict[str, dict] | None = None, max_enumerated: int = 3) -> list[Scenario]:
    """Enumerate plays/out for the most minutes-relevant uncertain players.

    Probabilities are attached only when every enumerated status has an empirical play-rate
    (see injuries.play_rates); otherwise prob=None for every scenario.
    PROBABLE players are treated as playing (flagged); lower-impact uncertain players beyond
    `max_enumerated` are assumed to play and flagged as PARTIAL.
    """
    unc = sorted(ctx.uncertain, key=lambda p: -baseline_minutes.get(p, 0.0))
    enum = unc[:max_enumerated]
    if len(unc) > max_enumerated:
        ctx.flags.append(f"UNCERTAIN_NOT_ENUMERATED:{','.join(unc[max_enumerated:])}")
    if not enum:
        return [Scenario("BASE", frozenset(), 1.0)]
    rates: dict[str, float | None] = {}
    for p in enum:
        r = (play_rates or {}).get(ctx.uncertain[p].value, {})
        rates[p] = r.get("p_play") if r else None
    have_probs = all(v is not None for v in rates.values())
    out = []
    for combo in itertools.product((True, False), repeat=len(enum)):
        extra = frozenset(p for p, plays in zip(enum, combo) if not plays)
        name = "+".join(f"{p}:{'PLAYS' if plays else 'OUT'}" for p, plays in zip(enum, combo))
        prob = None
        if have_probs:
            prob = 1.0
            for p, plays in zip(enum, combo):
                prob *= rates[p] if plays else 1 - rates[p]
        out.append(Scenario(name, extra, prob))
    return out
