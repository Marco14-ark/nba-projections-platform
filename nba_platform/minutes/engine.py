"""NBA_MINUTES_ENGINE_V1 — probabilistic player minutes conditioned on today's context.

Per player (fitted as of a date, on strictly earlier games only):

  minutes_i | plays = b_start*S + b_bench*(1-S)
                    + BX * (s_start*S + s_bench*(1-S))         # coach blowout behaviour
                    + sum_j  d_ij * absent_j                      # who absorbs whose minutes
                    + noise(sigma_i)

  * Coefficients are a recency-weighted Bayesian ridge regression with *informative prior means*:
      - b_start/b_bench: the player's own history (any team), shrunk to league role levels
      - s_start/s_bench: that coach's pooled blowout response (within-player), shrunk to league
      - d_ij: structural prior (teammate j's minutes x positional similarity x cap room), which
        the player's actual with/without games override as evidence accumulates.
    Absences seen in the data are learned; absences never seen fall back to the prior.
  * P(DNP | active) is modelled separately (bench only), by rotation-absence context.
  * Return from injury and new-team (trade) situations widen uncertainty with factors learned
    from league history when enough events exist, otherwise flagged UNCALIBRATED.

Simulation (per team, shared game-level draws across both teams):
  margin -> BX (blowout excess) and overtime count -> raw player minutes -> DNP draws ->
  projection onto the exact team constraint sum = 240 + 25*OT with 0 <= m_i <= cap_i,
  adjusting each player in proportion to his own predictive variance (Gaussian conditioning
  on the sum). OUT players are excluded before simulation, so they always get 0.

The Phase-1 margin distribution is an explicit placeholder (recency-weighted point-differential
ratings + home edge, all estimated from data, never from sportsbook lines). Phase 3 replaces it
with the game-markets score distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core.hashing import sha

ENGINE_NAME = "NBA_MINUTES_ENGINE_V1"
ENGINE_VERSION = "1.2.0"   # 1.2.0 (competition DNP model) promoted 2026-11-13 over 1.1.0; 1.1.0 over 1.0.0


@dataclass(frozen=True)
class MinutesConfig:
    half_life_games: float = 12.0
    lookback_games: int = 82
    prior_season_weight: float = 0.35
    blowout_threshold: float = 12.0
    prior_k_intercept: float = 3.0      # pseudo-games for intercept priors
    prior_k_absent: float = 3.0         # pseudo-games for with/without coefficients
    prior_k_blowout: float = 25.0
    prior_k_sigma: float = 5.0
    prior_k_dnp: float = 4.0
    max_absent_features: int = 8
    min_absent_games: float = 1.0       # weighted games needed on each side to learn d_ij
    struct_room_cap: float = 40.0
    new_team_sigma_c: float = 1.0       # sigma *= 1 + c/(1+n_games_new_team) (uncalibrated prior)
    return_streak_min: int = 3
    played_floor: float = 0.5
    # --- switches. Defaults = CHAMPION 1.1.0 (was CHALLENGER_B). 1.0.0 = ("room", "all", True) ---
    struct_weighting: str = "minutes_room"  # absorbers weighted by current role x remaining room
    absent_scope: str = "recent"            # only teammates seen in the last N team games count as absent
    absent_recent_games: int = 10
    dnp_context_buckets: bool = False       # pooled bench P(DNP) with recency weights
    prior_k_dnp_pooled: float = 2.0
    # --- V2 challenger switches (defaults keep champion 1.1.0 behaviour) ---
    ot_gamma: float = 0.0                 # >0: OT minutes go to players in proportion to (regulation share)^gamma
    ot_regulation_training: bool = False  # train on regulation-equivalent minutes (OT games rescaled)
    close_feature: bool = False           # learn extra minutes for starters/bench in close games (centred)
    close_threshold: float = 8.0
    dnp_model: str = "competition"        # CHAMPION 1.2.0. "pooled" = 1.1.0 behaviour
    sigma_dof_correct: bool = False       # residual variance / (1 - p_eff/n_eff): ridge hat-matrix trace + Kish n_eff
    sigma_dof_scope: str = "all"          # "all" | "starters"
    blowout_feature: str = "bx"           # "bx" (champion: max(0,|margin|-12)) | "garbage": PBP garbage-time minutes,
                                          #   drawn pregame from G | simulated |margin| (training games only)
    dnp_garbage: bool = False             # bench P(DNP) also depends on garbage time (fringe players enter blowouts)
    noise_model: str = "normal"           # "normal" | "empirical": standardized residuals observed in training
    # ---- V10 challengers (all off = champion 1.2.0) ----
    start_mix: bool = False               # MIN_V2_ROLE: unconfirmed lineups -> starting status drawn per simulation
    start_mix_power: float = 2.0          #   Plackett-Luce weights = start_score ** power (Gumbel top-5)
    proj_weight: str = "var"              # MIN_V2_JOINT: "var" (champion) | "var_role": residual minutes follow
                                          #   variance x expected role (rotation absorbs, deep bench does not)
    return_model: str = "ramp"            # MIN_V2_RETURN: "ramp" (champion) | "role_decay": ratio learned by
                                          #   games missed (3-5, 6-14, 15+) and prior role, own history only
                                          #   (keeps the real skew/left tail: foul trouble, in-game injury, benching)
    position_similarity: tuple = (("G", "G", 1.0), ("F", "F", 1.0), ("C", "C", 1.0),
                                  ("G", "F", 0.5), ("F", "C", 0.5), ("G", "C", 0.15))

    _V10_DEFAULTS = {"start_mix": False, "start_mix_power": 2.0, "proj_weight": "var", "return_model": "ramp"}

    def digest(self) -> str:
        d = asdict(self)
        for k, v in self._V10_DEFAULTS.items():      # champion fingerprint stays byte-identical
            if d.get(k) == v:
                d.pop(k, None)
        return sha(d)


def _season(d: str) -> str:
    y, m = int(d[:4]), int(d[5:7])
    s = y if m >= 8 else y - 1
    return f"{s}"


def _pos_bucket(p: str | None) -> str:
    if not p:
        return "?"
    p = p.upper()
    return "C" if p.startswith("C") else ("F" if p.startswith("F") else ("G" if p.startswith("G") else "?"))


@dataclass
class PlayerParams:
    player_id: str
    started: bool
    a: float                 # mean minutes at BX=0
    b: float                 # d minutes / d BX
    sigma: float
    c: float = 0.0           # d minutes / d (centred closeness)
    p_dnp: float = 0.0
    flags: list[str] = field(default_factory=list)
    proj_scale: float = 1.0  # V10 MIN_V2_RETURN: projection weight scale (uncertainty widening does not absorb cuts)


@dataclass
class TeamMinutes:
    team_id: int
    players: list[str]
    starters: list[str]
    samples: np.ndarray          # (n_sims, n_players)
    totals: np.ndarray           # (n_sims,) target total per sim
    params: dict[str, PlayerParams]

    def summary(self) -> dict[str, dict]:
        out = {}
        for k, pid in enumerate(self.players):
            x = self.samples[:, k]
            q = np.percentile(x, [10, 25, 50, 75, 90])
            p_dnp = float(np.mean(x <= 0.0))
            med = float(q[2])
            tier = "CORE" if med >= 28 and p_dnp < 0.05 else ("ROTATION" if med >= 15 and p_dnp < 0.25 else "FRINGE")
            out[pid] = {"mean": float(x.mean()), "median": med, "p10": float(q[0]), "p25": float(q[1]),
                        "p75": float(q[3]), "p90": float(q[4]), "p_dnp": p_dnp, "role_tier": tier,
                        "started": pid in self.starters, "flags": list(self.params[pid].flags)}
        return out


class MinutesEngine:
    NAME = ENGINE_NAME
    VERSION = ENGINE_VERSION

    def __init__(self, logs: pd.DataFrame, games: pd.DataFrame, as_of_date: str,
                 cfg: MinutesConfig | None = None, positions: dict[str, str] | None = None):
        self.cfg = cfg or MinutesConfig()
        self.as_of = as_of_date
        if len(logs) and (logs["game_date_et"] >= as_of_date).any():
            raise ValueError("leakage guard: logs contain games on/after as_of_date")
        if len(games) and (games["game_date_et"] >= as_of_date).any():
            games = games[games["game_date_et"] < as_of_date]
        self.positions = {k: _pos_bucket(v) for k, v in (positions or {}).items()}
        self.season = _season(as_of_date)
        lg = logs.copy()
        reason = lg["not_playing_reason"].fillna("").astype(str)
        lg["active"] = ~reason.str.startswith("INACTIVE")
        lg["started"] = lg["started"].astype(int)
        lg["played"] = lg["played"].astype(int) & lg["active"].astype(int)
        lg["minutes"] = lg["minutes"].astype(float)
        self.logs = lg
        self.games = games.copy()
        self._index()
        self._league_priors()
        self._team_blowout_slopes()
        self._fit_v2_priors()
        self._fit_noise_pools()
        self._return_ramp()
        self._margin_model()

    # ------------------------------------------------------------------ preparation
    def _index(self) -> None:
        lg = self.logs
        tg = lg[["team_id", "game_id", "game_date_et"]].drop_duplicates().sort_values(["team_id", "game_date_et", "game_id"])
        self.team_games: dict[int, list[str]] = {int(t): list(g["game_id"]) for t, g in tg.groupby("team_id")}
        self.team_age: dict[tuple[int, str], int] = {}
        self.game_season: dict[str, str] = {}
        for t, gids in self.team_games.items():
            n = len(gids)
            for k, gid in enumerate(gids):
                self.team_age[(t, gid)] = n - 1 - k
        for gid, d in tg[["game_id", "game_date_et"]].drop_duplicates().itertuples(index=False):
            self.game_season[gid] = _season(d)
        # margins per (game, team)
        self.margin: dict[tuple[str, int], float] = {}
        self.periods: dict[str, int] = {}
        for r in self.games.itertuples(index=False):
            if pd.isna(r.home_score) or pd.isna(r.away_score):
                continue
            m = float(r.home_score) - float(r.away_score)
            self.margin[(r.game_id, int(r.home_team_id))] = m
            self.margin[(r.game_id, int(r.away_team_id))] = -m
            self.periods[r.game_id] = int(r.periods) if not pd.isna(r.periods) else 4
        thr = self.cfg.blowout_threshold
        self.garbage_sampler = None
        gmap = {}
        if self.cfg.blowout_feature == "garbage" or self.cfg.dnp_garbage:
            if "garbage_min" not in self.games.columns:
                raise ValueError("blowout_feature='garbage' needs games['garbage_min'] (from PBP segments)")
            gg = self.games.dropna(subset=["garbage_min", "home_score", "away_score"])
            gmap = dict(zip(gg["game_id"], gg["garbage_min"].astype(float)))
            from .garbage import GarbageSampler
            reg = gg[gg["periods"].fillna(4) <= 4] if "periods" in gg.columns else gg
            self.garbage_sampler = GarbageSampler((reg["home_score"] - reg["away_score"]).abs().to_numpy(),
                                                  reg["garbage_min"].to_numpy())
        lg["gx"] = [gmap.get(g, 0.0) if self.periods.get(g, 4) <= 4 else 0.0 for g in lg["game_id"]]
        if self.cfg.blowout_feature == "garbage":
            lg["bx"] = lg["gx"]
        else:
            lg["bx"] = [max(0.0, abs(self.margin.get((g, int(t)), 0.0)) - thr) if self.periods.get(g, 4) <= 4 else 0.0
                        for g, t in zip(lg["game_id"], lg["team_id"])]
        ct = self.cfg.close_threshold
        lg["cx"] = [max(0.0, ct - abs(self.margin.get((g, int(t)), 0.0))) if self.periods.get(g, 4) <= 4 else ct
                    for g, t in zip(lg["game_id"], lg["team_id"])]
        if self.cfg.ot_regulation_training:
            per = np.array([self.periods.get(g, 4) for g in lg["game_id"]])
            lg["minutes"] = lg["minutes"] * 240.0 / (240.0 + 25.0 * np.maximum(per - 4, 0))
        lg["age"] = [self.team_age.get((int(t), g), 10_000) for g, t in zip(lg["game_id"], lg["team_id"])]
        hl = self.cfg.half_life_games
        w = np.power(0.5, lg["age"].to_numpy() / hl)
        same = np.array([self.game_season.get(g) == self.season for g in lg["game_id"]])
        w = np.where(same, w, w * self.cfg.prior_season_weight)
        w = np.where(lg["age"].to_numpy() < self.cfg.lookback_games, w, 0.0)
        lg["w"] = w
        # absence matrix per team: set of inactive player ids per game, and set of present ids
        self.inactive: dict[str, set[str]] = {}
        self.present: dict[str, set[str]] = {}
        for gid, g in lg.groupby("game_id"):
            self.inactive[gid] = set(g.loc[~g["active"], "player_id"])
            self.present[gid] = set(g.loc[g["active"], "player_id"])
        self.by_player = {pid: g for pid, g in lg.groupby("player_id")}

    def _league_priors(self) -> None:
        lg = self.logs
        played = lg[lg["played"] == 1]
        st, bn = played[played["started"] == 1], played[played["started"] == 0]
        self.mu_start = float(st["minutes"].mean()) if len(st) else 30.0
        self.mu_bench = float(bn["minutes"].mean()) if len(bn) else 16.0
        self.mu_all = float(played["minutes"].mean()) if len(played) else 20.0
        # within-player starter/bench gap (players with both)
        gaps, wts = [], []
        for pid, g in played.groupby("player_id"):
            s, b = g[g.started == 1]["minutes"], g[g.started == 0]["minutes"]
            if len(s) >= 2 and len(b) >= 2:
                gaps.append(s.mean() - b.mean())
                wts.append(min(len(s), len(b)))
        self.gap = float(np.average(gaps, weights=wts)) if gaps else self.mu_start - self.mu_bench
        # pooled residual sigma by role (deviation from player-role mean)
        def pooled_sd(df: pd.DataFrame) -> float:
            if len(df) < 10:
                return 6.0
            dev = df["minutes"] - df.groupby("player_id")["minutes"].transform("mean")
            return float(np.sqrt(np.mean(dev ** 2)) * np.sqrt(len(df) / max(1, len(df) - df["player_id"].nunique())))
        self.sd_start, self.sd_bench = pooled_sd(st), pooled_sd(bn)
        self.flags_global: list[str] = []
        if len(played) < 200:
            self.flags_global.append("LOW_TRAINING_VOLUME")
        # league bench DNP rate by absence context bucket
        act_bench = lg[(lg["active"]) & (lg["started"] == 0)]
        self.rot_players = self._rotation_players()
        ctx = np.array([self._absent_rotation_count(g, int(t)) > 0 for g, t in zip(act_bench["game_id"], act_bench["team_id"])])
        dnp = (act_bench["played"] == 0).to_numpy()
        self.dnp_pool_prior = float(dnp.mean()) if len(dnp) else 0.3
        self.dnp_prior = {False: float(dnp[~ctx].mean()) if (~ctx).sum() else 0.15,
                          True: float(dnp[ctx].mean()) if ctx.sum() else 0.10}

    def _rotation_players(self) -> dict[int, dict[str, float]]:
        """Weighted mean minutes when played, per team, for teammates (used for structural priors)."""
        out: dict[int, dict[str, float]] = {}
        p = self.logs[self.logs["played"] == 1]
        for (t, pid), g in p.groupby(["team_id", "player_id"]):
            w = g["w"].to_numpy()
            if w.sum() <= 0:
                continue
            out.setdefault(int(t), {})[pid] = float(np.average(g["minutes"], weights=w))
        return out

    def _absent_rotation_count(self, gid: str, team_id: int) -> int:
        rot = self.rot_players.get(team_id, {}) if hasattr(self, "rot_players") else {}
        return sum(1 for p in self.inactive.get(gid, ()) if rot.get(p, 0) >= 15)

    def _team_blowout_slopes(self) -> None:
        """Coach behaviour: within-player slope of minutes on BX, starters vs bench, per team."""
        p = self.logs[self.logs["played"] == 1].copy()
        p["dev"] = p["minutes"] - p.groupby(["player_id", "started"])["minutes"].transform("mean")
        p["bxd"] = p["bx"] - p.groupby(["player_id", "started"])["bx"].transform("mean")

        def slope(df: pd.DataFrame) -> tuple[float, float]:
            sxx = float((df["w"] * df["bxd"] ** 2).sum())
            sxy = float((df["w"] * df["bxd"] * df["dev"]).sum())
            return (sxy / sxx if sxx > 0 else 0.0), sxx
        self.league_slope = {s: slope(p[p.started == s])[0] for s in (0, 1)}
        self.team_slope: dict[tuple[int, int], float] = {}
        k = self.cfg.prior_k_blowout
        for (t, s), g in p.groupby(["team_id", "started"]):
            b, sxx = slope(g)
            # shrink: precision-weighted toward league (k in units of BX^2-weighted obs * 10)
            lam = k * 10.0
            self.team_slope[(int(t), int(s))] = (b * sxx + self.league_slope[int(s)] * lam) / (sxx + lam)

    def _return_ramp(self) -> None:
        ratios = {"SHORT": [], "LONG": []}
        for pid, g in self.by_player.items():
            for t, gt in g.groupby("team_id"):
                seq = gt.sort_values("age", ascending=False)  # oldest first
                streak, hist = 0, []
                for r in seq.itertuples(index=False):
                    if not r.active:
                        streak += 1
                        continue
                    if r.played and streak >= self.cfg.return_streak_min and len(hist) >= 3:
                        base = float(np.mean(hist[-10:]))
                        if base >= 10:
                            ratios["LONG" if streak >= 10 else "SHORT"].append(r.minutes / base)
                    if r.played:
                        hist.append(r.minutes)
                    streak = 0
        self.ramp: dict[str, tuple[float, float, bool]] = {}
        for b, arr in ratios.items():
            n = len(arr)
            if n >= 10:
                med = float(np.median(arr))
                k = 5.0
                self.ramp[b] = ((n * med + k) / (n + k), 1.0 + min(0.6, float(np.std(arr))), True)
            else:
                self.ramp[b] = (1.0, 1.25, False)

    def _margin_model(self) -> None:
        g = self.games.dropna(subset=["home_score", "away_score"]).copy()
        self.margin_flags: list[str] = []
        if len(g) < 30:
            self.home_edge, self.sd_margin = 0.0, 13.0
            self.margin_flags.append("UNCALIBRATED_MARGIN_PRIOR")
        g["m"] = g["home_score"].astype(float) - g["away_score"].astype(float)
        ratings: dict[int, float] = {}
        for t in set(g["home_team_id"]).union(g["away_team_id"]):
            rows = [(r.game_date_et, r.m if r.home_team_id == t else -r.m) for r in g.itertuples()
                    if t in (r.home_team_id, r.away_team_id)]
            rows.sort()
            if not rows:
                continue
            ms = np.array([m for _, m in rows])
            w = np.power(0.5, np.arange(len(ms))[::-1] / 20.0)
            ratings[int(t)] = float((w * ms).sum() / (w.sum() + 5.0))
        self.ratings = ratings
        if len(g) >= 30:
            self.home_edge = float(g["m"].mean())
            pred = np.array([ratings.get(int(h), 0) - ratings.get(int(a), 0) + self.home_edge
                             for h, a in zip(g["home_team_id"], g["away_team_id"])])
            self.sd_margin = float(np.std(g["m"].to_numpy() - pred))
        per = g["periods"].fillna(4).astype(int)
        n = len(per)
        if n >= 100:
            self.p_ot = float((per > 4).mean())
            n_ot = int((per > 4).sum())
            self.p_ot_cont = float((per > 5).sum() / n_ot) if n_ot >= 10 else 0.1
        else:
            self.p_ot, self.p_ot_cont = 0.06, 0.1
            self.margin_flags.append("UNCALIBRATED_OT_PRIOR")

    # ------------------------------------------------------------------ per-player model
    def start_scores(self, team_id: int, players: list[str]) -> dict[str, float]:
        """Recency-weighted start propensity. When regular starters are unavailable today, the
        rate is conditioned on past games in which those same starters were also missing — i.e.
        who this coach actually started in their place."""
        avail = set(players)
        regular = [p for p in self.rot_players.get(team_id, {}) if p not in avail and self._start_rate(p, team_id) >= 0.5]
        cond_games = {gid for gid in self.team_games.get(team_id, [])
                      if any(p in self.inactive.get(gid, set()) for p in regular)}
        # regular starters whose absence was never observed: fall back to positional similarity
        unseen = [j for j in regular if not any(j in self.inactive.get(gid, set())
                                                 for gid in self.team_games.get(team_id, []))]
        rot = self.rot_players.get(team_id, {})
        out = {}
        for pid in players:
            base = self._start_rate(pid, team_id)
            g = self.by_player.get(pid)
            if cond_games and g is not None:
                c = g[(g["team_id"] == team_id) & g["active"] & g["game_id"].isin(cond_games)]
                if c["w"].sum() > 0:
                    cr = float((c["w"] * c["started"]).sum() / (c["w"].sum() + 0.5))
                    base = 0.75 * cr + 0.25 * base
            if base < 0.5:
                for j in unseen:
                    base += 0.3 * self._similarity(pid, j) * min(1.0, rot.get(pid, 0.0) / 20.0)
            out[pid] = base
        return out

    def _start_rate(self, pid: str, team_id: int) -> float:
        g = self.by_player.get(pid)
        if g is None or not len(g):
            return 0.0
        a = g[g["active"]]
        on = a[a["team_id"] == team_id]
        if on["w"].sum() > 0:
            return float((on["w"] * on["started"]).sum() / (on["w"].sum() + 1.0))
        if a["w"].sum() > 0:
            return 0.5 * float((a["w"] * a["started"]).sum() / (a["w"].sum() + 1.0))
        return 0.0

    def _similarity(self, i: str, j: str) -> float:
        pi, pj = self.positions.get(i, "?"), self.positions.get(j, "?")
        if "?" in (pi, pj):
            return 0.5
        for a, b, s in self.cfg.position_similarity:
            if {a, b} == {pi, pj}:
                return s
        return 0.15

    def _base_level(self, pid: str) -> tuple[float, float, int]:
        """Player overall weighted played mean (any team), start rate, n played (all teams)."""
        g = self.by_player.get(pid)
        if g is None:
            return self.mu_bench, 0.0, 0
        p = g[(g["played"] == 1)]
        # recency within the player's own career sequence (handles trades)
        if not len(p):
            return self.mu_bench, 0.0, 0
        own_w = np.power(0.5, np.arange(len(p))[::-1] / self.cfg.half_life_games)
        order = p.sort_values("game_date_et")
        k = self.cfg.prior_k_intercept
        m = float((own_w * order["minutes"].to_numpy()).sum() + k * self.mu_all) / (own_w.sum() + k)
        sr = float((own_w * order["started"].to_numpy()).sum() / (own_w.sum() + 1.0))
        return m, sr, len(p)

    def player_params(self, pid: str, team_id: int, available: list[str], started: bool,
                      cap: float | None = None) -> PlayerParams:
        cfg = self.cfg
        flags: list[str] = []
        g = self.by_player.get(pid)
        rows = g[(g["team_id"] == team_id) & (g["active"]) & (g["w"] > 0)] if g is not None else pd.DataFrame()
        played = rows[rows["played"] == 1] if len(rows) else rows
        absent_today = [p for p in self._absent_scope(team_id) if p not in available and p != pid]
        rot = self.rot_players.get(team_id, {})

        m_i, sr_i, n_all = self._base_level(pid)
        mu_s = m_i + (1 - sr_i) * self.gap
        mu_b = m_i - sr_i * self.gap
        if n_all == 0:
            flags.append("NO_HISTORY")

        # structural prior for each teammate absence
        room = {k: max(0.0, cfg.struct_room_cap - rot.get(k, self.mu_bench)) for k in available}

        if cfg.struct_weighting == "minutes_room":
            # absorbers are weighted by their current role (minutes) x remaining room: rotation
            # players absorb, deep bench mostly does not
            absorb = {k: rot.get(k, 0.0) * room[k] / cfg.struct_room_cap for k in available}
        else:
            absorb = room

        def struct(j: str) -> float:
            m_j = rot.get(j, 0.0)
            den = sum(self._similarity(k, j) * absorb[k] for k in available if k != j)
            return m_j * self._similarity(pid, j) * absorb.get(pid, 0.0) / den if den > 0 else 0.0

        # choose absence features with variation in this player's own rows
        cands = sorted(set(absent_today) | set(sorted(rot, key=lambda k: -rot[k])[:cfg.max_absent_features]))
        cands = [j for j in cands if j != pid]
        feats, priors, fixed_adj = [], [], 0.0
        if len(played):
            w = played["w"].to_numpy()
            for j in cands:
                a = np.array([j not in self.present.get(gid, set()) for gid in played["game_id"]], dtype=float)
                wa, wp = float((w * a).sum()), float((w * (1 - a)).sum())
                if wa >= cfg.min_absent_games and wp >= cfg.min_absent_games:
                    feats.append((j, a))
                    priors.append(struct(j) if j in absent_today else struct(j))
                elif j in absent_today and wa < cfg.min_absent_games:
                    fixed_adj += struct(j)          # never/rarely seen without j: prior only
                    flags.append(f"UNSEEN_ABSENCE:{j}")
            feats = feats[:cfg.max_absent_features + len(absent_today)]
        else:
            for j in absent_today:
                fixed_adj += struct(j)

        a_coef, b_coef, sigma = None, None, None
        slope_s = self.team_slope.get((team_id, 1), self.league_slope[1])
        slope_b = self.team_slope.get((team_id, 0), self.league_slope[0])
        if len(played):
            S = played["started"].to_numpy().astype(float)
            BX = played["bx"].to_numpy()
            ncl = 0
            cols = [S, 1 - S, BX * S, BX * (1 - S)]
            mu_l = [mu_s, mu_b, slope_s, slope_b]
            lam_l = [cfg.prior_k_intercept] * 2 + [cfg.prior_k_blowout] * 2
            if cfg.close_feature:
                CX = played["cx"].to_numpy() - self.cx_mean
                cols += [CX * S, CX * (1 - S)]
                mu_l += [self.close_slope[1], self.close_slope[0]]
                lam_l += [cfg.prior_k_blowout] * 2
                ncl = 2
            cols += [a for _, a in feats]
            X = np.column_stack(cols)
            y = played["minutes"].to_numpy()
            w = played["w"].to_numpy()
            mu = np.array(mu_l + priors[:len(feats)])
            lam = np.array(lam_l + [cfg.prior_k_absent] * len(feats))
            XtW = X.T * w
            beta = np.linalg.solve(XtW @ X + np.diag(lam), XtW @ y + lam * mu)
            resid = y - X @ beta
            sd_pool = self.sd_start if started else self.sd_bench
            k = cfg.prior_k_sigma
            rss = (w * resid ** 2).sum()
            if cfg.sigma_dof_correct and (cfg.sigma_dof_scope == "all" or started):
                M = XtW @ X + np.diag(lam)
                p_eff = float(np.trace(np.linalg.solve(M, XtW @ X)))
                n_eff = float(w.sum() ** 2 / (w ** 2).sum())
                rss = rss / max(0.25, 1.0 - p_eff / max(n_eff, 1.0))
            sigma = float(np.sqrt((rss + k * sd_pool ** 2) / (w.sum() + k)))
            today_abs = np.array([1.0 if j in absent_today else 0.0 for j, _ in feats])
            base = beta[0] if started else beta[1]
            a_coef = float(base + (beta[4 + ncl:] @ today_abs if len(feats) else 0.0) + fixed_adj)
            b_coef = float(beta[2] if started else beta[3])
            c_coef = float((beta[4] if started else beta[5]) if ncl else 0.0)
        else:
            a_coef = (mu_s if started else mu_b) + fixed_adj
            b_coef = slope_s if started else slope_b
            c_coef = (self.close_slope[1] if started else self.close_slope[0]) if cfg.close_feature else 0.0
            sigma = (self.sd_start if started else self.sd_bench) * (1.3 if n_all == 0 else 1.0)

        # new team (trade / signing): fewer games with this team -> wider
        n_team = int(len(played))
        if g is not None and (g["team_id"] != team_id).any() and n_team < 15:
            sigma *= 1 + cfg.new_team_sigma_c / (1 + n_team)
            flags.append("NEW_TEAM")

        # return from injury
        ret_proj = 1.0
        streak = self._inactive_streak(pid, team_id)
        if streak >= cfg.return_streak_min:
            bucket = "LONG" if streak >= 10 else "SHORT"
            ramp, infl, calibrated = self.ramp[bucket]
            a_coef *= ramp
            sigma *= infl
            flags.append(f"RETURN_FROM_INJURY:{streak}g" + ("" if calibrated else ":UNCALIBRATED"))
            if cfg.return_model == "role_decay":
                ret_proj = 1.0 / infl

        # DNP probability (bench only)
        if started:
            p_dnp = 0.0
        elif cfg.dnp_model == "competition" and self.dnp_logit is not None:
            own = self._own_dnp_rate(rows)
            rank = sum(1 for k in available if k != pid and rot.get(k, 0.0) > rot.get(pid, 0.0))
            z = self.dnp_logit @ np.array([1.0, np.log(own / (1 - own)), rank, len(available)])
            p_dnp = float(1 / (1 + np.exp(-z)))
        elif not cfg.dnp_context_buckets:
            prior = self.dnp_pool_prior
            if len(rows):
                sel = rows["started"].to_numpy() == 0
                w = rows["w"].to_numpy()[sel]
                d = (rows["played"].to_numpy()[sel] == 0).astype(float)
                k = cfg.prior_k_dnp_pooled
                p_dnp = float(((w * d).sum() + k * prior) / (w.sum() + k))
            else:
                p_dnp = prior
        else:
            bucket = any(rot.get(j, 0) >= 15 for j in absent_today)
            prior = self.dnp_prior[bucket]
            if len(rows):
                ctx = np.array([self._absent_rotation_count(gid, team_id) > 0 for gid in rows["game_id"]])
                sel = (ctx == bucket) & (rows["started"].to_numpy() == 0)
                w = rows["w"].to_numpy()[sel]
                d = (rows["played"].to_numpy()[sel] == 0).astype(float)
                k = cfg.prior_k_dnp
                p_dnp = float(((w * d).sum() + k * prior) / (w.sum() + k))
            else:
                p_dnp = prior
        a_coef = max(0.0, a_coef)
        if cap is not None:
            flags.append(f"MINUTES_CAP:{cap:g}")
        return PlayerParams(pid, started, a_coef, b_coef, max(sigma, 0.5), c_coef, min(max(p_dnp, 0.0), 0.99), flags,
                            ret_proj)

    def _fit_noise_pools(self) -> None:
        """Standardized within-player residual pools (starters / bench) from training games only."""
        self.noise_pool = None
        if self.cfg.noise_model != "empirical":
            return
        p = self.logs[(self.logs["played"] == 1) & (self.logs["w"] > 0)]
        pools = {}
        for s_ in (0, 1):
            q = p[p["started"] == s_]
            g = q.groupby("player_id")["minutes"]
            n = g.transform("size")
            mu, sd = g.transform("mean"), g.transform("std")
            z = ((q["minutes"] - mu) / sd)[(n >= 8) & (sd > 0.5)]
            z = z.clip(-6, 6).to_numpy()
            z = (z - z.mean()) / z.std() if len(z) > 200 else np.random.default_rng(0).standard_normal(1000)
            pools[s_] = z
        self.noise_pool = pools

    def _own_dnp_rate(self, rows: pd.DataFrame) -> float:
        prior, k = self.dnp_pool_prior, self.cfg.prior_k_dnp_pooled
        if not len(rows):
            return prior
        sel = rows["started"].to_numpy() == 0
        w = rows["w"].to_numpy()[sel]
        d = (rows["played"].to_numpy()[sel] == 0).astype(float)
        return float(np.clip(((w * d).sum() + k * prior) / (w.sum() + k), 0.01, 0.99))

    def _fit_v2_priors(self) -> None:
        """League-level pieces for V2 switches (fitted on training rows only)."""
        lg = self.logs
        p = lg[(lg["played"] == 1) & (lg["w"] > 0)]
        self.cx_mean = float(np.average(p["cx"], weights=p["w"])) if len(p) else 0.0
        self.close_slope = {0: 0.0, 1: 0.0}
        if self.cfg.close_feature and len(p):
            for s_ in (0, 1):
                q = p[p["started"] == s_]
                dev = q["minutes"] - q.groupby("player_id")["minutes"].transform("mean")
                x = q["cx"] - q.groupby("player_id")["cx"].transform("mean")
                sxx = float((q["w"] * x * x).sum())
                self.close_slope[s_] = float((q["w"] * x * dev).sum() / sxx) if sxx > 0 else 0.0
        self.dnp_logit = None
        self.dnp_garbage_coef = 0.0
        if self.cfg.dnp_model == "competition":
            a = lg[lg["active"] & (lg["started"] == 0) & (lg["w"] > 0)].copy()
            if len(a) < 500:
                return
            own = {}
            for pid, g in a.groupby("player_id"):
                own[pid] = self._own_dnp_rate(g)
            rot_by_team = self.rot_players
            act = lg[lg["active"]]
            n_act = act.groupby(["game_id", "team_id"]).size()
            ranks = []
            for r in a.itertuples(index=False):
                rot = rot_by_team.get(int(r.team_id), {})
                mates = self.present.get(r.game_id, set())
                mine = rot.get(r.player_id, 0.0)
                ranks.append(sum(1 for k in mates if k != r.player_id and rot.get(k, 0.0) > mine))
            o = np.array([own[p_] for p_ in a["player_id"]])
            cols = [np.ones(len(a)), np.log(o / (1 - o)), ranks,
                    [n_act.get((g, t), 13) for g, t in zip(a["game_id"], a["team_id"])]]
            if self.cfg.dnp_garbage:
                cols.append(a["gx"].to_numpy())
            X = np.column_stack(cols)
            y = (a["played"].to_numpy() == 0).astype(float)
            w = a["w"].to_numpy()
            kdim = X.shape[1]
            beta = np.zeros(kdim)
            for _ in range(25):                      # weighted IRLS logistic regression + tiny ridge
                eta = X @ beta
                mu = 1 / (1 + np.exp(-eta))
                W = w * mu * (1 - mu) + 1e-9
                H = (X.T * W) @ X + 1e-3 * np.eye(kdim)
                beta = beta + np.linalg.solve(H, X.T @ (w * (y - mu)) - 1e-3 * beta)
            self.dnp_logit = beta[:4]
            self.dnp_garbage_coef = float(beta[4]) if self.cfg.dnp_garbage else 0.0

    def _absent_scope(self, team_id: int) -> list[str]:
        rot = self.rot_players.get(team_id, {})
        if self.cfg.absent_scope != "recent":
            return list(rot)
        recent = self.team_games.get(team_id, [])[-self.cfg.absent_recent_games:]
        seen = set().union(*[self.present.get(g, set()) | self.inactive.get(g, set()) for g in recent]) if recent else set()
        return [p for p in rot if p in seen]

    def _inactive_streak(self, pid: str, team_id: int) -> int:
        gids = self.team_games.get(team_id, [])
        streak = 0
        for gid in reversed(gids):
            if pid in self.inactive.get(gid, set()):
                streak += 1
            elif pid in self.present.get(gid, set()):
                break
            else:
                break   # not on team then
        return streak

    # ------------------------------------------------------------------ simulation
    def margin_draws(self, home_id: int, away_id: int, n: int, seed: int, return_margin: bool = False):
        rng = np.random.default_rng([seed, 0])
        mu = self.ratings.get(home_id, 0.0) - self.ratings.get(away_id, 0.0) + self.home_edge
        m = rng.normal(mu, self.sd_margin, n)
        tau = self.sd_margin * norm.ppf(0.5 + self.p_ot / 2)
        tie = np.abs(m) < tau
        n_ot = np.where(tie, rng.geometric(1 - self.p_ot_cont, n), 0)
        bx = self.bx_from_margin(np.where(tie, 0.0, m), n_ot, seed)
        if return_margin:
            return bx, n_ot, np.where(tie, 0.0, m)
        return bx, n_ot

    def bx_from_margin(self, margin: np.ndarray, n_ot: np.ndarray, seed: int) -> np.ndarray:
        """Blowout feature for each simulated game: champion = max(0,|m|-thr); challenger = a garbage-time
        draw from historical games with a similar final margin. Uses only the SIMULATED margin."""
        margin = np.asarray(margin, float)
        if self.cfg.blowout_feature == "garbage":
            return self.garbage_sampler.draw(margin, n_ot, np.random.default_rng([seed, 99]))
        return np.where(np.asarray(n_ot) > 0, 0.0, np.maximum(0.0, np.abs(margin) - self.cfg.blowout_threshold))

    def garbage_for_dnp(self, margin: np.ndarray | None, n_ot: np.ndarray, bx: np.ndarray, seed: int) -> np.ndarray:
        if self.cfg.blowout_feature == "garbage":
            return bx
        if margin is None or self.garbage_sampler is None:
            return np.zeros(len(bx))
        # same draw for both teams of a game: seed from |margin| (identical for home and away)
        key = int(np.abs(np.asarray(margin, float)).sum() * 1000) % (2 ** 31)
        return self.garbage_sampler.draw(margin, n_ot, np.random.default_rng([key, 99]))

    def simulate_team(self, team_id: int, available: list[str], starters: list[str],
                      caps: dict[str, float], bx: np.ndarray, n_ot: np.ndarray, seed: int,
                      margin: np.ndarray | None = None, starters_confirmed: bool = False) -> TeamMinutes:
        n = len(bx)
        players = list(available)
        params = {p: self.player_params(p, team_id, available, p in starters, caps.get(p)) for p in players}
        rng = np.random.default_rng([seed, team_id % 100_000])
        A = np.array([params[p].a for p in players])
        B = np.array([params[p].b for p in players])
        SD = np.array([params[p].sigma for p in players])
        PD = np.array([params[p].p_dnp for p in players])
        smask = None
        if self.cfg.start_mix and not starters_confirmed and len(players) > 5:
            # MIN_V2_ROLE: the starting five is uncertain until confirmed. Draw it per simulation
            # (Plackett-Luce via Gumbel top-5 on start propensities) and use the matching start/bench
            # parameters, instead of a single deterministic expected lineup.
            sc = self.start_scores(team_id, players)
            w = np.array([max(sc.get(p, 0.0), 1e-3) for p in players]) ** self.cfg.start_mix_power
            rs = np.random.default_rng([seed, team_id % 100_000, 7])
            g = np.log(w)[None, :] + rs.gumbel(size=(n, len(players)))
            top = np.argpartition(-g, 4, axis=1)[:, :5]
            smask = np.zeros((n, len(players)), dtype=bool)
            np.put_along_axis(smask, top, True, axis=1)
            alt = {p: self.player_params(p, team_id, available, p not in starters, caps.get(p)) for p in players}
            ps = {p: (params[p] if p in starters else alt[p]) for p in players}
            pb = {p: (alt[p] if p in starters else params[p]) for p in players}
            A = np.where(smask, np.array([ps[p].a for p in players]), np.array([pb[p].a for p in players]))
            B = np.where(smask, np.array([ps[p].b for p in players]), np.array([pb[p].b for p in players]))
            SD = np.where(smask, np.array([ps[p].sigma for p in players]), np.array([pb[p].sigma for p in players]))
            PD = np.where(smask, 0.0, np.array([pb[p].p_dnp for p in players]))
        total = 240.0 + 25.0 * n_ot
        per_cap = 48.0 + 5.0 * n_ot
        cap = np.minimum(per_cap[:, None], np.array([caps.get(p, np.inf) for p in players])[None, :])
        Cc = np.array([params[p].c for p in players])
        if self.cfg.close_feature and margin is not None:
            cx = np.where(n_ot > 0, self.cfg.close_threshold,
                          np.maximum(0.0, self.cfg.close_threshold - np.abs(margin))) - self.cx_mean
        else:
            cx = np.zeros(n)
        A2 = A if A.ndim == 2 else A[None, :]
        B2 = B if B.ndim == 2 else B[None, :]
        reg = A2 + B2 * bx[:, None] + Cc[None, :] * cx[:, None]
        if self.cfg.ot_gamma > 0:
            share = np.power(np.clip(A2.mean(0), 0, None) / 48.0, self.cfg.ot_gamma)
            share = share / share.sum() if share.sum() > 0 else np.full(len(A), 1 / len(A))
            raw = reg + (25.0 * n_ot)[:, None] * share[None, :]
        else:
            raw = reg * (total / 240.0)[:, None]
        if self.noise_pool is not None:
            Z = np.empty((n, len(players)))
            for k_, p_ in enumerate(players):
                pool = self.noise_pool[1 if p_ in starters else 0]
                Z[:, k_] = pool[rng.integers(0, len(pool), n)]
        else:
            Z = rng.standard_normal((n, len(players)))
        SD2 = SD if SD.ndim == 2 else SD[None, :]
        raw = raw + SD2 * Z
        if self.cfg.dnp_garbage and self.dnp_garbage_coef != 0.0:
            gdraw = self.garbage_for_dnp(margin, n_ot, bx, seed)
            PD2 = PD if PD.ndim == 2 else PD[None, :]
            lp = np.log(np.clip(PD2, 1e-6, 1 - 1e-6) / (1 - np.clip(PD2, 1e-6, 1 - 1e-6)))
            PDn = 1 / (1 + np.exp(-(lp + self.dnp_garbage_coef * gdraw[:, None])))
            PDn = np.where(PD2 > 0, PDn, 0.0)          # starters keep P(DNP)=0
            plays = rng.random((n, len(players))) >= PDn
        else:
            plays = rng.random((n, len(players))) >= (PD if PD.ndim == 2 else PD[None, :])
        var = SD2 ** 2 * (np.array([params[p].proj_scale for p in players]) ** 2)[None, :]
        if self.cfg.proj_weight == "var_role":
            # MIN_V2_JOINT: residual minutes (budget = 240 + OT) are absorbed in proportion to
            # variance x expected regulation role, so deep-bench noise does not soak up rotation minutes
            var = var * np.clip(A2, 1.0, None) / 24.0
        x = project_minutes(raw, plays, cap, total, var, self.cfg.played_floor)
        return TeamMinutes(team_id, players, list(starters), x, total, params)

    def fingerprint(self) -> str:
        return sha({"name": self.NAME, "v": self.VERSION, "cfg": self.cfg.digest(), "as_of": self.as_of,
                    "n_logs": int(len(self.logs))})


def project_minutes(raw: np.ndarray, plays: np.ndarray, cap: np.ndarray, total: np.ndarray,
                    var: np.ndarray, floor: float = 0.5, iters: int = 60) -> np.ndarray:
    """Project each simulated row onto {sum = total, floor <= x_i <= cap_i for players, 0 otherwise}.

    Ensures feasibility first (activates DNP players with the highest raw minutes if the playing
    set cannot reach the total), then iteratively distributes the residual in proportion to each
    free player's variance, clipping at bounds.
    """
    n, k = raw.shape
    plays = plays.copy()
    cap = np.broadcast_to(cap, (n, k)).copy()
    # feasibility: need >=5 players and sum of caps >= total
    for _ in range(k):
        need = (plays.sum(1) < 5) | ((cap * plays).sum(1) < total - 1e-9)
        if not need.any():
            break
        cand = np.where(plays, -np.inf, raw)
        j = np.argmax(cand, axis=1)
        rows = np.where(need)[0]
        plays[rows, j[rows]] = True
    lo = np.where(plays, np.minimum(floor, cap), 0.0)
    hi = np.where(plays, cap, 0.0)
    x = np.clip(np.where(plays, raw, 0.0), lo, hi)
    v = np.broadcast_to(var, (n, k))
    for _ in range(iters):
        r = total - x.sum(1)
        if np.max(np.abs(r)) < 1e-9:
            break
        up = r > 0
        free = np.where(up[:, None], x < hi - 1e-12, x > lo + 1e-12) & plays
        wts = np.where(free, v, 0.0)
        s = wts.sum(1)
        s = np.where(s > 0, s, 1.0)
        x = np.clip(x + r[:, None] * wts / s[:, None], lo, hi)
    return x
