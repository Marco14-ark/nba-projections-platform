"""Walk-forward validation for minutes (spec §47-49, §88, §90).

For each test date D: fit on games strictly before D, predict every active player on D, score
against actual minutes. Two information settings:
  * CONFIRMED: actual starters known (proxy for the T-30m window, after lineups are announced)
  * EXPECTED:  starters predicted by the model (proxy for earlier windows)
Availability uses the official game-day inactive list (known before tip).

Metrics: MAE (mean & median as point forecast), bias, RMSE, CRPS, P10-P90 / P25-P75 coverage,
broken down by starter status, predicted role tier and injury context.
Champion/challenger comparison: paired per-row absolute-error differences with a game-cluster
bootstrap CI. Promotion requires the whole CI below zero and no coverage degradation > 5 pts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .engine import MinutesConfig, MinutesEngine
from ..lineups.resolve import expected_starters


def crps_samples(x: np.ndarray, y: float) -> float:
    xs = np.sort(x)
    n = len(xs)
    term1 = np.mean(np.abs(xs - y))
    term2 = (2.0 / (n * n)) * np.sum((2 * np.arange(1, n + 1) - n - 1) * xs)
    return float(term1 - 0.5 * term2)


@dataclass
class BacktestResult:
    name: str
    rows: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _metrics(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"n": 0}
    e = df["mean"] - df["actual"]
    return {"n": int(len(df)), "mae": float(np.mean(np.abs(e))),
            "mae_median": float(np.mean(np.abs(df["median"] - df["actual"]))),
            "bias": float(np.mean(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "crps": float(df["crps"].mean()),
            "cov80": float(np.mean((df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"]))),
            "cov50": float(np.mean((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"])))}


def summarize(rows: pd.DataFrame) -> dict:
    out = {"ALL": _metrics(rows)}
    for col in ("started", "tier", "injury_ctx"):
        if col in rows:
            for k, g in rows.groupby(col):
                out[f"{col}={k}"] = _metrics(g)
    return out


def _day_games(logs: pd.DataFrame, d: str) -> list[tuple[str, int, list[str], list[str], dict]]:
    out = []
    day = logs[logs["game_date_et"] == d]
    for (gid, tid), g in day.groupby(["game_id", "team_id"]):
        active = g[~g["not_playing_reason"].fillna("").astype(str).str.startswith("INACTIVE")]
        out.append((gid, int(tid), list(active["player_id"]), list(active.loc[active["started"] == 1, "player_id"]),
                    dict(zip(active["player_id"], active["minutes"].astype(float)))))
    return out


def backtest_engine(logs: pd.DataFrame, games: pd.DataFrame, dates: list[str], cfg: MinutesConfig | None = None,
                    n_sims: int = 2000, seed: int = 11, starters_mode: str = "CONFIRMED",
                    positions: dict | None = None, name: str = "CHAMPION") -> BacktestResult:
    recs = []
    for d in dates:
        eng = MinutesEngine(logs[logs["game_date_et"] < d], games[games["game_date_et"] < d], d, cfg, positions)
        day_games = games[games["game_date_et"] == d]
        for gid, tid, active, starters_actual, actual in _day_games(logs, d):
            gr = day_games[day_games["game_id"] == gid]
            if not len(gr):
                continue
            h, a = int(gr.iloc[0]["home_team_id"]), int(gr.iloc[0]["away_team_id"])
            bx, n_ot, mg = eng.margin_draws(h, a, n_sims, seed, return_margin=True)
            starters = (starters_actual if starters_mode == "CONFIRMED"
                        else expected_starters(active, eng.start_scores(tid, active)))
            tm = eng.simulate_team(tid, active, starters, {}, bx, n_ot, seed, margin=mg if tid == h else -mg)
            summ = tm.summary()
            rot = eng.rot_players.get(tid, {})
            scope = set(eng._absent_scope(tid)) if hasattr(eng, "_absent_scope") else set(rot)
            inj = any(rot.get(p, 0) >= 20 and p not in active and p in scope for p in rot)
            for k, p in enumerate(tm.players):
                s = summ[p]
                recs.append({"date": d, "game_id": gid, "team_id": tid, "player_id": p, "actual": actual[p],
                             "started": int(p in starters_actual), "tier": s["role_tier"],
                             "injury_ctx": int(inj), "mean": s["mean"], "median": s["median"],
                             "p10": s["p10"], "p25": s["p25"], "p75": s["p75"], "p90": s["p90"],
                             "crps": crps_samples(tm.samples[::max(1, n_sims // 1000), k], actual[p]),
                             "flags": ";".join(s["flags"])})
    rows = pd.DataFrame(recs)
    return BacktestResult(name, rows, summarize(rows))


def backtest_baseline(logs: pd.DataFrame, dates: list[str], last_n: int = 10, name: str = "BASELINE_LAST10",
                      include_dnp: bool = False) -> BacktestResult:
    """Naive baseline: mean/quantiles of the player's last N played games (any team).
    include_dnp=True: last N games in which he was AVAILABLE, counting DNPs as 0 (stronger)."""
    recs = []
    reason = logs["not_playing_reason"].fillna("").astype(str)
    played = logs[~reason.str.startswith("INACTIVE")] if include_dnp else \
        logs[(logs["played"] == 1) & ~reason.str.startswith("INACTIVE")]
    league_bench = float(played.loc[played["started"] == 0, "minutes"].mean()) if len(played) else 15.0
    for d in dates:
        hist = played[played["game_date_et"] < d].sort_values("game_date_et")
        last = {p: g["minutes"].to_numpy()[-last_n:] for p, g in hist.groupby("player_id")}
        for gid, tid, active, starters_actual, actual in _day_games(logs, d):
            for p in active:
                x = last.get(p)
                if x is None or len(x) < 3:
                    x = np.array([league_bench])
                q = np.percentile(x, [10, 25, 50, 75, 90])
                recs.append({"date": d, "game_id": gid, "team_id": tid, "player_id": p, "actual": actual[p],
                             "started": int(p in starters_actual), "mean": float(np.mean(x)),
                             "median": float(q[2]), "p10": q[0], "p25": q[1], "p75": q[3], "p90": q[4],
                             "crps": crps_samples(x.astype(float), actual[p])})
    rows = pd.DataFrame(recs)
    return BacktestResult(name, rows, summarize(rows))


def _cluster_ci(d: np.ndarray, games: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    ug, inv = np.unique(games, return_inverse=True)
    S = np.bincount(inv, weights=d)
    C = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    boots = S[idx].sum(1) / C[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def compare(champion: BacktestResult, challenger: BacktestResult, n_boot: int = 2000, seed: int = 3,
            coverage_tolerance: float = 0.05, primary: str = "crps") -> dict:
    """Paired comparison on identical rows. Negative diff = challenger better.

    primary='crps' (default, the right score for a distributional forecast) or 'mae'.
    PROMOTE: primary CI entirely < 0, the other metric not significantly worse, and 80% interval
    coverage not further from nominal than the champion's by more than `coverage_tolerance`.
    """
    key = ["game_id", "player_id"]
    m = champion.rows[key + ["mean", "crps", "actual"]].merge(
        challenger.rows[key + ["mean", "crps"]], on=key, suffixes=("_c", "_x"))
    games = m["game_id"].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = {"mae": (np.abs(m["mean_x"] - m["actual"]) - np.abs(m["mean_c"] - m["actual"])).to_numpy(),
             "crps": (m["crps_x"] - m["crps_c"]).to_numpy()}
    res = {"n_rows": int(len(m)), "n_games": int(len(np.unique(games))), "primary": primary}
    for k, d in diffs.items():
        res[f"{k}_diff"] = float(d.mean())
        res[f"{k}_ci95"] = list(_cluster_ci(d, games, n_boot, rng))
    other = "mae" if primary == "crps" else "crps"
    cov_c = champion.metrics["ALL"]["cov80"]
    cov_x = challenger.metrics["ALL"]["cov80"]
    cov_ok = abs(cov_x - 0.8) <= abs(cov_c - 0.8) + coverage_tolerance
    lo, hi = res[f"{primary}_ci95"]
    if hi < 0 and cov_ok and res[f"{other}_ci95"][0] <= 0:
        decision = "PROMOTE"
    elif lo > 0:
        decision = "REJECT"
    else:
        decision = "KEEP_CHAMPION"
    res.update({"cov80_champion": cov_c, "cov80_challenger": cov_x, "decision": decision})
    return res
