"""Game-level minutes forecast: lineup context + availability scenarios + engine simulation."""
from __future__ import annotations

import numpy as np

from ..lineups.resolve import Scenario, TeamContext, build_scenarios, expected_starters
from .engine import MinutesEngine, TeamMinutes


def _summ(samples: np.ndarray, players: list[str]) -> dict[str, dict]:
    out = {}
    for k, p in enumerate(players):
        x = samples[:, k]
        q = np.percentile(x, [10, 25, 50, 75, 90])
        out[p] = {"mean": float(x.mean()), "median": float(q[2]), "p10": float(q[0]), "p25": float(q[1]),
                  "p75": float(q[3]), "p90": float(q[4]), "p_dnp": float(np.mean(x <= 0))}
    return out


def forecast_team(engine: MinutesEngine, ctx: TeamContext, bx: np.ndarray, n_ot: np.ndarray, seed: int,
                  play_rates: dict | None = None) -> dict:
    baseline = engine.rot_players.get(ctx.team_id, {})
    scenarios = build_scenarios(ctx, baseline, play_rates)
    runs: list[tuple[Scenario, TeamMinutes]] = []
    for sc in scenarios:
        avail = ctx.available(sc)
        starters = expected_starters(avail, engine.start_scores(ctx.team_id, avail), ctx.confirmed_starters)
        tm = engine.simulate_team(ctx.team_id, avail, starters, ctx.minutes_caps, bx, n_ot, seed)
        runs.append((sc, tm))
    out = {"team_id": ctx.team_id, "lineup_state": ctx.lineup_state.value, "flags": list(ctx.flags),
           "out": sorted(ctx.out), "uncertain": {p: s.value for p, s in ctx.uncertain.items()},
           "scenarios": [{"name": sc.name, "prob": sc.prob, "starters": tm.starters,
                          "players": tm.summary()} for sc, tm in runs],
           "mixture": None}
    if len(runs) > 1 and all(sc.prob is not None for sc, _ in runs):
        n = len(bx)
        rng = np.random.default_rng([seed, 7, ctx.team_id % 100_000])
        probs = np.array([sc.prob for sc, _ in runs])
        pick = rng.choice(len(runs), size=n, p=probs / probs.sum())
        allp = sorted({p for _, tm in runs for p in tm.players})
        mix = np.zeros((n, len(allp)))
        for r, (_, tm) in enumerate(runs):
            rows = pick == r
            for k, p in enumerate(tm.players):
                mix[rows, allp.index(p)] = tm.samples[rows, k]
        out["mixture"] = _summ(mix, allp)
    elif len(runs) > 1:
        out["flags"].append("SCENARIOS_WITHOUT_PROBABILITIES")
    return out


def forecast_game(engine: MinutesEngine, home: TeamContext, away: TeamContext, home_id: int, away_id: int,
                  n_sims: int = 10_000, seed: int = 0, play_rates: dict | None = None) -> dict:
    bx, n_ot = engine.margin_draws(home_id, away_id, n_sims, seed)
    return {"engine": engine.NAME, "engine_version": engine.VERSION, "engine_fingerprint": engine.fingerprint(),
            "n_sims": n_sims, "seed": seed, "p_ot": float(np.mean(n_ot > 0)),
            "p_blowout": float(np.mean(bx > 0)), "margin_flags": list(engine.margin_flags),
            "global_flags": list(engine.flags_global),
            "home": forecast_team(engine, home, bx, n_ot, seed, play_rates),
            "away": forecast_team(engine, away, bx, n_ot, seed, play_rates)}
