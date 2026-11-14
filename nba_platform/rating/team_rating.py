"""NBA_TEAM_RATING_V2 — injury-aware team strength from players actually available.

Champion (V1): recency-weighted point-differential team ratings + home edge (team level; blind
to who plays tonight).

Challenger (V2, this module):
  1. Player offensive / defensive impact per 100 possessions from lineup-vs-lineup segments
     reconstructed from play-by-play (regularised adjusted plus-minus, recency-weighted, ridge
     shrinkage toward 0 = replacement-average). Opponent strength is built in: every segment
     controls for the five opponents on the floor.
  2. Tonight's team offence/defence = sum over AVAILABLE players of (expected minutes / 48) x
     impact, so injuries and the quality of replacements change the rating directly.
  3. Home court and rest (back-to-back, rest days) are measured on training games, not assumed.
  4. Possessions from the pace model; margin & total = possessions x rating difference / sum.
All inputs are from games strictly before the forecast date.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..core.hashing import sha

RATING_NAME = "NBA_TEAM_RATING_V2"


@dataclass(frozen=True)
class RatingConfig:
    ridge: float = 3000.0            # possessions-equivalent shrinkage toward 0
    half_life_days: float = 90.0
    prior_season_weight: float = 0.5
    min_poss_segment: int = 1

    def digest(self) -> str:
        return sha(asdict(self))


def _season_start(d: str) -> int:
    y, m = int(d[:4]), int(d[5:7])
    return y if m >= 8 else y - 1


class PointDiffRating:
    """Champion V1 margin model (identical logic to MinutesEngine._margin_model)."""

    def __init__(self, games: pd.DataFrame, as_of: str):
        g = games[(games["game_date_et"] < as_of)].dropna(subset=["home_score", "away_score"]).copy()
        g["m"] = g["home_score"].astype(float) - g["away_score"].astype(float)
        self.ratings = {}
        for t in set(g["home_team_id"]).union(g["away_team_id"]):
            rows = sorted((r.game_date_et, r.m if r.home_team_id == t else -r.m) for r in g.itertuples()
                          if t in (r.home_team_id, r.away_team_id))
            ms = np.array([m for _, m in rows])
            w = np.power(0.5, np.arange(len(ms))[::-1] / 20.0)
            self.ratings[int(t)] = float((w * ms).sum() / (w.sum() + 5.0))
        self.home_edge = float(g["m"].mean()) if len(g) >= 30 else 0.0
        pred = np.array([self.ratings.get(int(h), 0) - self.ratings.get(int(a), 0) + self.home_edge
                         for h, a in zip(g["home_team_id"], g["away_team_id"])])
        self.sd = float(np.std(g["m"].to_numpy() - pred)) if len(g) >= 30 else 13.0

    def margin(self, home: int, away: int) -> float:
        return self.ratings.get(home, 0.0) - self.ratings.get(away, 0.0) + self.home_edge


def rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (pregame-known schedule information)."""
    games = games[(games["home_team_id"] > 0) & (games["away_team_id"] > 0)]
    long = pd.concat([games[["game_id", "game_date_et", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                      games[["game_id", "game_date_et", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    long = long.sort_values(["team_id", "game_date_et"])
    d = pd.to_datetime(long["game_date_et"])
    long["rest"] = d.groupby(long["team_id"]).diff().dt.days.fillna(7).clip(upper=7).to_numpy()
    return long[["game_id", "team_id", "rest"]].drop_duplicates(["game_id", "team_id"])


class LineupRating:
    NAME = RATING_NAME

    def __init__(self, segments: pd.DataFrame, games: pd.DataFrame, as_of: str, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.as_of = as_of
        g = games[games["game_date_et"] < as_of]
        seg = segments.merge(g[["game_id", "game_date_et"]], on="game_id")
        if (seg["game_date_et"] >= as_of).any():
            raise ValueError("leakage guard")
        age = (pd.Timestamp(as_of) - pd.to_datetime(seg["game_date_et"])).dt.days.to_numpy()
        w = np.power(0.5, age / self.cfg.half_life_days)
        season = _season_start(as_of)
        old = np.array([_season_start(d) != season for d in seg["game_date_et"]])
        w = np.where(old, w * self.cfg.prior_season_weight, w)
        self._fit(seg, w)
        self._fit_context(g)

    def _fit(self, seg: pd.DataFrame, w: np.ndarray) -> None:
        players = sorted(set(",".join(seg["home5"]).split(",")) | set(",".join(seg["away5"]).split(",")))
        players = [p for p in players if p]
        ix = {p: i for i, p in enumerate(players)}
        P = len(players)
        rows, cols, vals, y, ww, home_flag = [], [], [], [], [], []
        r = 0
        for h5, a5, hp, ap, hq, aq, wt in zip(seg["home5"], seg["away5"], seg["home_pts"], seg["away_pts"],
                                              seg["home_poss"], seg["away_poss"], w):
            hs, as_ = h5.split(","), a5.split(",")
            for off, de, pts, poss, hf in ((hs, as_, hp, hq, 1.0), (as_, hs, ap, aq, 0.0)):
                if poss < self.cfg.min_poss_segment:
                    continue
                for p in off:
                    rows.append(r); cols.append(ix[p]); vals.append(1.0)
                for p in de:
                    rows.append(r); cols.append(P + ix[p]); vals.append(1.0)
                rows.append(r); cols.append(2 * P); vals.append(hf)          # home offence
                y.append(100.0 * pts / poss)
                ww.append(poss * wt)
                r += 1
        X = csr_matrix((vals, (rows, cols)), shape=(r, 2 * P + 1))
        y, ww = np.array(y), np.array(ww)
        self.mean_ortg = float(np.average(y, weights=ww))
        yc = y - self.mean_ortg
        XtW = X.T.multiply(ww).tocsr()
        A = (XtW @ X).toarray()
        lam = np.full(2 * P + 1, self.cfg.ridge)
        lam[-1] = 1.0                                                          # home term ~unpenalised
        beta = np.linalg.solve(A + np.diag(lam), XtW @ yc)
        self.players = ix
        self.off = beta[:P]
        self.deff = beta[P:2 * P]
        self.home_off = float(beta[-1])
        self.n_poss = float(ww.sum())

    def player_impact(self, pid: str) -> tuple[float, float]:
        i = self.players.get(pid)
        return (0.0, 0.0) if i is None else (float(self.off[i]), float(self.deff[i]))

    def team_ratings(self, minutes: dict[str, float]) -> tuple[float, float]:
        """(offence, defence) per 100 for a set of available players with expected minutes."""
        tot = sum(minutes.values())
        scale = 240.0 / tot if tot > 0 else 0.0
        o = d = 0.0
        for p, m in minutes.items():
            oi, di = self.player_impact(p)
            o += (m * scale / 48.0) * oi
            d += (m * scale / 48.0) * di
        return o, d

    def _fit_context(self, g: pd.DataFrame) -> None:
        """Rest effect on margin, measured on training games (residual vs neutral expectation)."""
        rd = rest_days(g)
        x = g.merge(rd.rename(columns={"team_id": "home_team_id", "rest": "rh"}), on=["game_id", "home_team_id"]) \
             .merge(rd.rename(columns={"team_id": "away_team_id", "rest": "ra"}), on=["game_id", "away_team_id"])
        x = x.dropna(subset=["home_score", "away_score"])
        m = (x["home_score"] - x["away_score"]).to_numpy(float)
        b2b = (x["rh"] <= 1).astype(float) - (x["ra"] <= 1).astype(float)
        if len(x) >= 200:
            Xc = np.column_stack([np.ones(len(x)), b2b])
            coef = np.linalg.lstsq(Xc, m, rcond=None)[0]
            self.b2b_effect = float(coef[1])          # margin change when home is on a B2B (vs away)
        else:
            self.b2b_effect = 0.0

    def predict(self, home_min: dict[str, float], away_min: dict[str, float], poss: float,
                home_rest: float = 2.0, away_rest: float = 2.0) -> dict:
        ho, hd = self.team_ratings(home_min)
        ao, ad = self.team_ratings(away_min)
        h_ortg = self.mean_ortg + self.home_off + ho + ad
        a_ortg = self.mean_ortg + ao + hd
        rest = self.b2b_effect * (float(home_rest <= 1) - float(away_rest <= 1))
        hp, ap = poss * h_ortg / 100.0, poss * a_ortg / 100.0
        return {"home_pts": hp + rest / 2, "away_pts": ap - rest / 2, "margin": hp - ap + rest, "total": hp + ap}
