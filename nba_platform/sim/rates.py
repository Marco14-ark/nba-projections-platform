"""NBA_OPPORTUNITY_RATES_V1 + NBA_EFFICIENCY_V1 — per-possession opportunity and conversion rates.

Separates OPPORTUNITY from EFFICIENCY (spec §1):

  opportunity (per team possession while ON COURT, from play-by-play on-court tracking)
    use_rate   (FGA + TOV + FT trips) / team possessions on court       ~ usage
    fga/tov/trip split of those uses
    p3         3PA / FGA            prim   rim attempts / 2PA
    oreb_rate  OREB / team missed FG while on court      ("rebound chances" estimator)
    dreb_rate  DREB / opponent missed FG while on court
    ast_rate   AST / teammate made FG while on court     ("potential assists" estimator)
  efficiency
    3P%, rim FG%, non-rim 2P%, FT%, FTA per trip

Free data has no tracking (touches, potential assists, rebound chances). The on-court
denominators above are ESTIMATORS built from observable events; they are labelled as such and
validated through the downstream stat forecasts.

All rates are empirical-Bayes shrunk (beta/gamma style, prior strength in units of the
denominator) toward league means, with recency weights over the player's own games.
Team rates (pace, uses per possession, rebounding, assisted share) and opponent-allowed ratios
(defensive matchup) are fitted the same way at team-game level and shrunk toward 1.0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..core.hashing import sha

RATES_NAME = "NBA_OPPORTUNITY_RATES_V1"
RATES_VERSION = "1.1.0"   # 1.1.0: hl 45 + league environment (promoted 2026-11-13)


@dataclass(frozen=True)
class RatesConfig:
    hl_player_games: float = 45.0      # 1.1.0 (1.0.0 = 20)
    hl_team_games: float = 25.0
    prior_season_weight: float = 0.5
    k_use: float = 150.0        # team possessions on court
    k_split: float = 60.0       # uses
    k_p3: float = 60.0          # FGA
    k_prim: float = 60.0        # 2PA
    k_3pct: float = 250.0       # 3PA  (3P% is very noisy: heavy shrinkage)
    k_rimpct: float = 120.0
    k_midpct: float = 150.0
    k_ftpct: float = 80.0
    k_reb: float = 150.0        # misses on court
    k_ast: float = 120.0        # teammate makes on court
    k_def_games: float = 20.0   # opponent-allowed ratios
    k_def_shoot_games: float = 40.0
    league_env: bool = True       # 1.1.0: current-season league pace/efficiency vs training average
    k_env_games: float = 60.0     # shrinkage (games) of the current-season league ratio toward 1
    # V9 challenger: "league" = champion (lg * h/lg * a/lg * env_pace); "team_mean" = h * a / mean(team
    # factors), no env_pace (team factors already carry the current season through recency weights).
    pace_norm: str = "league"

    def digest(self) -> str:
        d = asdict(self)
        if d["pace_norm"] == "league":          # keep the champion fingerprint byte-identical
            del d["pace_norm"]
        return sha(d)


def _season_start(d: str) -> int:
    y, m = int(d[:4]), int(d[5:7])
    return y if m >= 8 else y - 1


def _eb(num: np.ndarray | float, den: np.ndarray | float, prior: float, k: float):
    return (num + k * prior) / (den + k)


class OpportunityRates:
    NAME = RATES_NAME
    VERSION = RATES_VERSION

    def __init__(self, logs: pd.DataFrame, games: pd.DataFrame, as_of: str, cfg: RatesConfig | None = None):
        self.cfg = cfg or RatesConfig()
        self.as_of = as_of
        if len(logs) and (logs["game_date_et"] >= as_of).any():
            raise ValueError("leakage guard: logs contain games on/after as_of")
        games = games[games["game_date_et"] < as_of]
        season = _season_start(as_of)
        p = logs[logs["played"] == 1].copy()
        p["mid_fgm"] = p["fgm"] - p["fg3m"] - p["rim_fgm"]
        p["fg2a"] = p["fga"] - p["fg3a"]
        p["uses"] = p["fga"] + p["tov"] + p["ft_trips"]
        p["tm_fgm_ex"] = p["team_fgm_on"] - p["fgm"]
        p = p.sort_values(["player_id", "game_date_et"])
        age = p.groupby("player_id").cumcount(ascending=False).to_numpy()
        w = np.power(0.5, age / self.cfg.hl_player_games)
        old = np.array([_season_start(d) != season for d in p["game_date_et"]])
        p["w"] = np.where(old, w * self.cfg.prior_season_weight, w)
        self._fit_league(p)
        self._fit_players(p)
        self._fit_teams(p, games, season)

    # --------------------------------------------------------------------------- league
    def _fit_league(self, p: pd.DataFrame) -> None:
        s = p[["uses", "fga", "tov", "ft_trips", "fg3a", "fg3m", "fg2a", "rim_fga", "rim_fgm", "mid_fga",
               "mid_fgm", "fta", "ftm", "oreb", "dreb", "ast", "team_poss_on", "team_miss_on", "opp_miss_on",
               "tm_fgm_ex"]].sum()
        self.L = {
            "use": s.uses / s.team_poss_on, "f_fga": s.fga / s.uses, "f_tov": s.tov / s.uses,
            "f_trip": s.ft_trips / s.uses, "p3": s.fg3a / s.fga, "prim": s.rim_fga / s.fg2a,
            "pct3": s.fg3m / s.fg3a, "pctrim": s.rim_fgm / s.rim_fga, "pctmid": s.mid_fgm / s.mid_fga,
            "pctft": s.ftm / s.fta, "fta_trip": s.fta / s.ft_trips, "oreb": s.oreb / s.team_miss_on,
            "dreb": s.dreb / s.opp_miss_on, "ast": s.ast / s.tm_fgm_ex,
        }

    # --------------------------------------------------------------------------- players
    def _fit_players(self, p: pd.DataFrame) -> None:
        cols = ["uses", "fga", "tov", "ft_trips", "fg3a", "fg3m", "fg2a", "rim_fga", "rim_fgm", "mid_fga",
                "mid_fgm", "fta", "ftm", "oreb", "dreb", "ast", "team_poss_on", "team_miss_on", "opp_miss_on",
                "tm_fgm_ex", "minutes"]
        W = p[cols].multiply(p["w"], axis=0)
        W["player_id"] = p["player_id"].to_numpy()
        S = W.groupby("player_id").sum()
        S["wsum"] = p.groupby("player_id")["w"].sum()
        S["n"] = p.groupby("player_id").size()
        c, L = self.cfg, self.L
        uses = S["uses"].clip(lower=0)
        split_tot = uses + c.k_split
        out = pd.DataFrame(index=S.index)
        out["use"] = _eb(uses, S["team_poss_on"], L["use"], c.k_use)
        out["f_fga"] = (S["fga"] + c.k_split * L["f_fga"]) / split_tot
        out["f_tov"] = (S["tov"] + c.k_split * L["f_tov"]) / split_tot
        out["f_trip"] = (S["ft_trips"] + c.k_split * L["f_trip"]) / split_tot
        out["p3"] = _eb(S["fg3a"], S["fga"], L["p3"], c.k_p3)
        out["prim"] = _eb(S["rim_fga"], S["fg2a"], L["prim"], c.k_prim)
        out["pct3"] = _eb(S["fg3m"], S["fg3a"], L["pct3"], c.k_3pct)
        out["pctrim"] = _eb(S["rim_fgm"], S["rim_fga"], L["pctrim"], c.k_rimpct)
        out["pctmid"] = _eb(S["mid_fgm"], S["mid_fga"], L["pctmid"], c.k_midpct)
        out["pctft"] = _eb(S["ftm"], S["fta"], L["pctft"], c.k_ftpct)
        out["fta_trip"] = _eb(S["fta"], S["ft_trips"], L["fta_trip"], 30.0)
        out["oreb"] = _eb(S["oreb"], S["team_miss_on"], L["oreb"], c.k_reb)
        out["dreb"] = _eb(S["dreb"], S["opp_miss_on"], L["dreb"], c.k_reb)
        out["ast"] = _eb(S["ast"], S["tm_fgm_ex"], L["ast"], c.k_ast)
        # effective sample sizes for uncertainty in skill (beta posterior strength)
        out["n3"] = S["fg3a"] + c.k_3pct
        out["nrim"] = S["rim_fga"] + c.k_rimpct
        out["nmid"] = S["mid_fga"] + c.k_midpct
        out["nft"] = S["fta"] + c.k_ftpct
        out["games"] = S["n"]
        self.P = out

    def player(self, pid: str) -> dict:
        if pid in self.P.index:
            r = self.P.loc[pid].to_dict()
            r["known"] = True
            return r
        L, c = self.L, self.cfg
        return {"use": L["use"] * 0.8, "f_fga": L["f_fga"], "f_tov": L["f_tov"], "f_trip": L["f_trip"],
                "p3": L["p3"], "prim": L["prim"], "pct3": L["pct3"], "pctrim": L["pctrim"], "pctmid": L["pctmid"],
                "pctft": L["pctft"], "fta_trip": L["fta_trip"], "oreb": L["oreb"], "dreb": L["dreb"], "ast": L["ast"],
                "n3": c.k_3pct, "nrim": c.k_rimpct, "nmid": c.k_midpct, "nft": c.k_ftpct, "games": 0, "known": False}

    # --------------------------------------------------------------------------- teams
    def _fit_teams(self, p: pd.DataFrame, games: pd.DataFrame, season: int) -> None:
        agg = ["uses", "fga", "fgm", "fg3a", "fg3m", "fg2a", "rim_fga", "rim_fgm", "mid_fga", "mid_fgm",
               "fta", "ftm", "ft_trips", "tov", "oreb", "dreb", "ast", "pts"]
        tg = p.groupby(["game_id", "team_id"])[agg].sum().reset_index()
        g = games[["game_id", "game_date_et", "home_team_id", "away_team_id", "home_poss", "away_poss", "periods"]]
        tg = tg.merge(g, on="game_id", how="inner")
        home = tg["team_id"] == tg["home_team_id"]
        tg["poss"] = np.where(home, tg["home_poss"], tg["away_poss"])
        tg["opp_id"] = np.where(home, tg["away_team_id"], tg["home_team_id"])
        tg["game_min"] = 48 + 5 * (tg["periods"] - 4)
        tg["pace48"] = tg["poss"] * 48 / tg["game_min"]
        opp = tg[["game_id", "team_id"] + agg].rename(columns={k: f"o_{k}" for k in agg + ["team_id"]})
        tg = tg.merge(opp, left_on=["game_id", "opp_id"], right_on=["game_id", "o_team_id"])
        tg = tg.sort_values(["team_id", "game_date_et"])
        age = tg.groupby("team_id").cumcount(ascending=False).to_numpy()
        w = np.power(0.5, age / self.cfg.hl_team_games)
        old = np.array([_season_start(d) != season for d in tg["game_date_et"]])
        tg["w"] = np.where(old, w * self.cfg.prior_season_weight, w)
        self.TG = tg
        L = self.L
        lg = {"pace48": float(tg["pace48"].mean()), "kappa": float((tg["uses"] / tg["poss"]).mean()),
              "oreb_t": float(tg["oreb"].sum() / (tg["fga"] - tg["fgm"]).sum()),
              "dreb_t": float(tg["dreb"].sum() / (tg["o_fga"] - tg["o_fgm"]).sum()),
              "ast_t": float(tg["ast"].sum() / tg["fgm"].sum())}
        self.LT = lg
        kd, ks = self.cfg.k_def_games, self.cfg.k_def_shoot_games

        def wavg(x, ww):
            return float((x * ww).sum() / ww.sum())

        rows = {}
        for t, d in tg.groupby("team_id"):
            ww = d["w"].to_numpy()
            n_eff = ww.sum()
            sh = lambda val, prior, k: (val * n_eff + prior * k) / (n_eff + k)  # noqa: E731
            r = {
                "pace48": sh(wavg(d["pace48"], ww), lg["pace48"], 10.0),
                "kappa": sh(wavg(d["uses"] / d["poss"], ww), lg["kappa"], 10.0),
                "oreb_t": sh(wavg(d["oreb"] / (d["fga"] - d["fgm"]).clip(lower=1), ww), lg["oreb_t"], kd),
                "dreb_t": sh(wavg(d["dreb"] / (d["o_fga"] - d["o_fgm"]).clip(lower=1), ww), lg["dreb_t"], kd),
                "ast_t": sh(wavg(d["ast"] / d["fgm"].clip(lower=1), ww), lg["ast_t"], kd),
                # defensive ratios: what opponents do against this team relative to league
                "d_p3": sh(wavg((d["o_fg3a"] / d["o_fga"]) / L["p3"], ww), 1.0, kd),
                "d_pct3": sh(wavg((d["o_fg3m"] / d["o_fg3a"].clip(lower=1)) / L["pct3"], ww), 1.0, ks),
                "d_pctrim": sh(wavg((d["o_rim_fgm"] / d["o_rim_fga"].clip(lower=1)) / L["pctrim"], ww), 1.0, ks),
                "d_pctmid": sh(wavg((d["o_mid_fgm"] / d["o_mid_fga"].clip(lower=1)) / L["pctmid"], ww), 1.0, ks),
                "d_trip": sh(wavg((d["o_ft_trips"] / d["o_uses"]) / L["f_trip"], ww), 1.0, kd),
                "d_tov": sh(wavg((d["o_tov"] / d["o_uses"]) / L["f_tov"], ww), 1.0, kd),
                "games": int(len(d)),
            }
            rows[int(t)] = r
        self.T = rows
        # home court (spec §33): measured, not assumed. Ratio of home to away points per possession.
        hp = tg[tg["team_id"] == tg["home_team_id"]]
        ap = tg[tg["team_id"] != tg["home_team_id"]]
        self._fit_score_residuals(games)
        self._fit_league_env(tg, season)
        if len(hp) >= 100:
            r = (hp["pts"].sum() / hp["poss"].sum()) / (ap["pts"].sum() / ap["poss"].sum())
            self.home_eff = float(np.sqrt(r))      # home x sqrt(r), away / sqrt(r)
        else:
            self.home_eff = 1.0
        # residual SD of game pace around the multiplicative pace model
        pred = np.array([self.expected_pace(int(h), int(a)) for h, a in zip(g["home_team_id"], g["away_team_id"])]) \
            if len(g) else np.array([])
        if len(g):
            actual = ((g["home_poss"] + g["away_poss"]) / 2 * 48 / (48 + 5 * (g["periods"] - 4))).to_numpy()
            self.pace_sd = float(np.std(actual - pred))
        else:
            self.pace_sd = 5.0

    def _fit_score_residuals(self, games: pd.DataFrame) -> None:
        """Predictive SDs of game margin and total, from an additive offence/defence/home model
        on regulation games (recency-weighted), inflated for degrees of freedom. These are the
        targets the joint simulation is calibrated to (spec §26: correlations from data)."""
        g = games[(games["periods"] == 4)].dropna(subset=["home_score", "away_score"])
        self.sd_margin_t, self.sd_total_t, self.score_corr = 14.0, 18.0, 0.2
        self.score_flags = []
        if len(g) < 300:
            self.score_flags.append("UNCALIBRATED_SCORE_VARIANCE")
            return
        teams = sorted(set(g["home_team_id"]) | set(g["away_team_id"]))
        ix = {t: i for i, t in enumerate(teams)}
        nt = len(teams)
        n = len(g)
        X = np.zeros((2 * n, 2 * nt + 1))
        y = np.concatenate([g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)])
        hi = np.array([ix[t] for t in g["home_team_id"]])
        ai = np.array([ix[t] for t in g["away_team_id"]])
        X[np.arange(n), hi] = 1
        X[np.arange(n), nt + ai] = 1
        X[np.arange(n), -1] = 1
        X[n + np.arange(n), ai] = 1
        X[n + np.arange(n), nt + hi] = 1
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        infl = np.sqrt(2 * n / (2 * n - X.shape[1]))
        rh, ra = res[:n] * infl, res[n:] * infl
        self.sd_margin_t = float(np.std(rh - ra))
        self.sd_total_t = float(np.std(rh + ra))
        self.score_corr = float(np.corrcoef(rh, ra)[0, 1])

    def _fit_league_env(self, tg: pd.DataFrame, season: int) -> None:
        """Season-to-season environment shift (e.g. 2026-27 scoring fell ~4 pts/game vs 2025-26)."""
        self.env_pace, self.env_eff, self.env_n = 1.0, 1.0, 0
        if not self.cfg.league_env or not len(tg):
            return
        cur = tg[np.array([_season_start(d) == season for d in tg["game_date_et"]])]
        n = len(cur) / 2.0                                   # team-games -> games
        self.env_n = int(n)
        if n == 0:
            return
        w = tg["w"].to_numpy()
        lg_pace = float(np.average(tg["pace48"], weights=w))
        lg_ppp = float((tg["pts"] * tg["w"]).sum() / (tg["poss"] * tg["w"]).sum())
        cur_pace = float(cur["pace48"].mean())
        cur_ppp = float(cur["pts"].sum() / cur["poss"].sum())
        k = self.cfg.k_env_games
        self.env_pace = float((n * cur_pace / lg_pace + k) / (n + k))
        self.env_eff = float((n * cur_ppp / lg_ppp + k) / (n + k))

    def team(self, tid: int) -> dict:
        if tid in self.T:
            return self.T[tid]
        lg = self.LT
        return {"pace48": lg["pace48"], "kappa": lg["kappa"], "oreb_t": lg["oreb_t"], "dreb_t": lg["dreb_t"],
                "ast_t": lg["ast_t"], "d_p3": 1.0, "d_pct3": 1.0, "d_pctrim": 1.0, "d_pctmid": 1.0, "d_trip": 1.0,
                "d_tov": 1.0, "games": 0}

    def expected_pace(self, home: int, away: int) -> float:
        if self.cfg.pace_norm == "team_mean" and self.T:
            lgw = float(np.mean([v["pace48"] for v in self.T.values()]))
            return self.team(home)["pace48"] * self.team(away)["pace48"] / lgw
        lg = self.LT["pace48"]
        return lg * (self.team(home)["pace48"] / lg) * (self.team(away)["pace48"] / lg) * getattr(self, "env_pace", 1.0)

    def fingerprint(self) -> str:
        return sha({"n": self.NAME, "v": self.VERSION, "cfg": self.cfg.digest(), "as_of": self.as_of,
                    "np": int(len(self.P))})
