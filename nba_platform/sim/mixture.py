"""Exact mixture of joint simulations across availability scenarios.

Each draw is assigned to one scenario with a seeded multinomial pick (weights frozen before the
game). A player missing from a scenario contributes zeros for that scenario's draws (he is OUT).
The result is a regular GameSim, so the same market functions summarise it.
"""
from __future__ import annotations

import numpy as np

from .game_sim import GameSim, TeamSim


def _mix_team(teams: list[TeamSim], pick: np.ndarray) -> TeamSim:
    players = sorted({p for t in teams for p in t.players})
    n = len(pick)
    keys = sorted(set.intersection(*[set(t.stats) for t in teams]))
    stats = {k: np.zeros((n, len(players)), dtype=teams[0].stats[k].dtype) for k in keys}
    score = np.zeros(n, dtype=teams[0].score.dtype)
    misses = np.zeros(n, dtype=np.asarray(teams[0].misses).dtype)
    fgm = np.zeros(n, dtype=np.asarray(teams[0].fgm).dtype)
    for r, t in enumerate(teams):
        rows = pick == r
        if not rows.any():
            continue
        cols = [players.index(p) for p in t.players]
        for k in keys:
            stats[k][np.ix_(rows, cols)] = t.stats[k][rows]
        score[rows] = t.score[rows]
        misses[rows] = np.asarray(t.misses)[rows]
        fgm[rows] = np.asarray(t.fgm)[rows]
    return TeamSim(teams[0].team_id, players, stats, score, misses, fgm)


def mix_game_sims(sims: list[GameSim], weights: list[float], seed: int) -> tuple[GameSim, np.ndarray]:
    if len(sims) != len(weights) or not sims:
        raise ValueError("sims/weights mismatch")
    n = len(sims[0].n_ot)
    if any(len(s.n_ot) != n for s in sims):
        raise ValueError("all scenario simulations must have the same number of draws")
    w = np.asarray(weights, float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid scenario weights")
    rng = np.random.default_rng([seed, 4242])
    pick = rng.choice(len(sims), size=n, p=w / w.sum())
    home = _mix_team([s.home for s in sims], pick)
    away = _mix_team([s.away for s in sims], pick)

    def cat(attr):
        out = np.array(getattr(sims[0], attr), copy=True)
        for r, s in enumerate(sims):
            out[pick == r] = np.asarray(getattr(s, attr))[pick == r]
        return out

    meta = {"mixture_of": len(sims), "weights": [float(x) for x in w / w.sum()],
            "draw_share": [float(np.mean(pick == r)) for r in range(len(sims))]}
    return GameSim(home, away, cat("n_ot"), cat("latent_margin"), cat("possessions"), meta), pick
