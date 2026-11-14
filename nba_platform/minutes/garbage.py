"""Garbage-time measurement from lineup segments (play-by-play) and pregame G|margin sampling.

Definition (fixed a priori, not tuned on outcomes): 4th-quarter game time during which the lead
at the start of the stint is >= 25 with 12-9 min left, >= 20 with 9-6 min left, >= 15 with
< 6 min left. OT is never garbage time. G is in minutes (0-12) and is the same for both teams.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q = 720.0


def threshold(remaining_s: np.ndarray) -> np.ndarray:
    r = np.asarray(remaining_s, float) / 60.0
    return np.where(r > 9, 25.0, np.where(r > 6, 20.0, 15.0))


def game_garbage(segments: pd.DataFrame) -> pd.DataFrame:
    """segments: game_id, period, start, end (s within period), home_pts, away_pts, home5, away5.
    Returns per game: garbage_min, lead_end_q3, starters' share of garbage on-court (diagnostic)."""
    out = []
    for gid, g in segments.sort_values(["game_id", "period", "start"]).groupby("game_id", sort=False):
        lead = (g["home_pts"] - g["away_pts"]).cumsum().to_numpy() - (g["home_pts"] - g["away_pts"]).to_numpy()
        q4 = (g["period"] == 4).to_numpy()
        st = g["start"].to_numpy()
        en = g["end"].to_numpy()
        G = 0.0
        if q4.any():
            # split each Q4 stint at the threshold breakpoints (3 and 6 min marks are at 180/360 s remaining)
            for a, b, l in zip(st[q4], en[q4], lead[q4]):
                cuts = sorted({a, b, *[x for x in (Q - 540, Q - 360) if a < x < b]})
                for x0, x1 in zip(cuts[:-1], cuts[1:]):
                    if abs(l) >= threshold(Q - x0):
                        G += (x1 - x0) / 60.0
        q3_end = lead[(g["period"] == 4).to_numpy()][0] if q4.any() else np.nan
        out.append({"game_id": gid, "garbage_min": round(G, 3), "lead_start_q4": q3_end})
    return pd.DataFrame(out)


class GarbageSampler:
    """Empirical G | |final margin| from training games only (bins), used inside the pregame
    simulation: the simulated margin -> a G draw from historical games with similar margins."""

    EDGES = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 200])

    def __init__(self, abs_margin: np.ndarray, garbage: np.ndarray, min_bin: int = 30):
        abs_margin = np.asarray(abs_margin, float)
        garbage = np.asarray(garbage, float)
        self.pools = []
        idx = np.digitize(abs_margin, self.EDGES[1:-1])
        for b in range(len(self.EDGES) - 1):
            pool = garbage[idx == b]
            lo = b
            while len(pool) < min_bin and lo > 0:          # widen downwards (never invent values)
                lo -= 1
                pool = garbage[(idx >= lo) & (idx <= b)]
            self.pools.append(pool if len(pool) else np.zeros(1))
        self.n_train = int(len(garbage))

    def draw(self, margin: np.ndarray, n_ot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        m = np.abs(np.asarray(margin, float))
        idx = np.digitize(m, self.EDGES[1:-1])
        out = np.zeros(len(m))
        for b, pool in enumerate(self.pools):
            rows = idx == b
            if rows.any():
                out[rows] = pool[rng.integers(0, len(pool), rows.sum())]
        return np.where(np.asarray(n_ot) > 0, 0.0, out)
