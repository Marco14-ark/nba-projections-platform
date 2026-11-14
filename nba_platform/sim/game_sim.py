"""NBA_FULL_PLAYER_MONTE_CARLO_V1 — one joint simulation per game; every stat shares the game state.

Causal order inside each simulation (spec §24), vectorised across simulations:

  AVAILABILITY (inputs) -> LATENT GAME SCRIPT (margin, OT) -> PLAYER MINUTES (minutes engine,
  conditioned on the script: blowouts pull starters, OT adds minutes) -> GAME POSSESSIONS (pace of
  both teams) -> TEAM USES (FGA+TOV+FT trips per possession) -> PLAYER USES (usage x on-court
  possessions; absent players' usage is absorbed by whoever is on the floor) -> FGA/TOV/FT-trip
  split (opponent forces TOV / fouls) -> shot type (3PA vs rim vs other 2PA; opponent shot
  profile) -> conversion (player skill drawn from its posterior x opponent defence x home court x
  game-script efficiency shock) -> free throws -> POINTS -> REBOUNDS from the missed shots that
  actually happened -> ASSISTS from teammates' made shots that actually happened -> TEAM SCORE.

Team accounting holds in every simulation by construction and is checked by `accounting_check`:
team points = sum of player points; offensive + opponent defensive rebounds <= missed FG;
each player's assists <= teammates' made FG; team minutes = 240 + 25*OT.

Coupling of script and score (avoids circularity): a pilot run with no blowout gives the
bottom-up margin distribution (mean mu, pure-simulation variance v0). The latent script margin
is drawn from N(mu, sigma_t) where sigma_t is the empirical margin SD; it sets blowout/OT for
minutes, and a team efficiency shock of size rho*(latent - mu) with rho = sqrt(1 - v0/sigma_t^2)
moves the simulated score toward it, so the final margin SD matches the data and big final
margins coincide with the sims in which starters sat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..minutes.engine import MinutesEngine
from .rates import OpportunityRates

SIM_NAME = "NBA_FULL_PLAYER_MONTE_CARLO_V1"
SIM_VERSION = "1.0.0"
STATS = ["minutes", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "reb", "ast", "tov"]


@dataclass
class TeamInput:
    team_id: int
    available: list[str]
    starters: list[str]
    caps: dict[str, float] = field(default_factory=dict)


@dataclass
class TeamSim:
    team_id: int
    players: list[str]
    stats: dict[str, np.ndarray]        # each (n, k)
    score: np.ndarray                   # (n,)
    misses: np.ndarray
    fgm: np.ndarray
    assist_clip: float = 0.0


@dataclass
class GameSim:
    home: TeamSim
    away: TeamSim
    n_ot: np.ndarray
    latent_margin: np.ndarray
    possessions: np.ndarray
    meta: dict

    @property
    def margin(self) -> np.ndarray:          # home - away
        return self.home.score - self.away.score

    @property
    def total(self) -> np.ndarray:
        return self.home.score + self.away.score


def _norm_rows(w: np.ndarray) -> np.ndarray:
    s = w.sum(-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    p = w / s
    p[..., -1] = np.clip(1.0 - p[..., :-1].sum(-1), 0.0, 1.0)
    return p


def _multinomial(rng: np.random.Generator, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    return rng.multinomial(n.astype(np.int64), _norm_rows(p))


@dataclass
class JointConfig:
    """JOINT_MC_V2 switches (all off = champion). Variances are ESTIMATED on a training season
    (scripts/estimate_joint_v2.py), never set to hit a target correlation.
      form_sd      (A) player game-state factor: one lognormal multiplier per player per draw on his
                       usage, rebounding and assist opportunity weights.
      team_use_sd  (B) team opportunity shock: uses per possession vary by team-game (offensive
                       rebounds / extra possessions), shared by all teammates.
      ast_on_makes (B) assist allocation proportional to the passer's rate x teammates' made shots
                       in that draw (a passer cannot assist his own basket)."""
    form_sd: float = 0.0
    team_use_sd: float = 0.0
    ast_on_makes: bool = False

    def tag(self) -> str:
        return f"f{self.form_sd:.3f}_t{self.team_use_sd:.3f}_a{int(self.ast_on_makes)}"


class GameSimulator:
    NAME = SIM_NAME
    VERSION = SIM_VERSION

    def __init__(self, minutes: MinutesEngine, rates: OpportunityRates, joint: JointConfig | None = None):
        self.m, self.r = minutes, rates
        self.j = joint or JointConfig()

    # ------------------------------------------------------------------ one team's offense
    def _team(self, rng, t: TeamInput, opp_id: int, M: np.ndarray, P: np.ndarray, gm: np.ndarray,
              eff: np.ndarray, home: bool):
        R = [self.r.player(p) for p in t.available]
        g = lambda k: np.array([x[k] for x in R])  # noqa: E731
        T, D = self.r.team(t.team_id), self.r.team(opp_id)
        n, k = M.shape
        poss_on = P[:, None] * M / gm[:, None]
        if self.j.team_use_sd > 0:
            tau = rng.normal(-0.5 * self.j.team_use_sd ** 2, self.j.team_use_sd, n)
            uses_T = np.round(P * T["kappa"] * np.exp(tau)).astype(np.int64)
        else:
            uses_T = np.round(P * T["kappa"]).astype(np.int64)   # a possession ends in ~kappa uses: not Poisson
        if self.j.form_sd > 0:
            sf = self.j.form_sd
            form = np.exp(rng.normal(-0.5 * sf ** 2, sf, (n, k)))
        else:
            form = np.ones((n, k))
        uses = _multinomial(rng, uses_T, g("use")[None, :] * poss_on * form)
        f_tov = g("f_tov") * D["d_tov"]
        f_trip = g("f_trip") * D["d_trip"]
        f_fga = np.clip(1.0 - f_tov - f_trip, 0.05, None)
        split = np.stack([f_fga, f_tov, f_trip], -1)                       # (k, 3)
        sp = rng.multinomial(uses, _norm_rows(np.broadcast_to(split, (n, k, 3)).copy()))
        fga, tov, trips = sp[..., 0], sp[..., 1], sp[..., 2]
        fg3a = rng.binomial(fga, np.clip(g("p3") * D["d_p3"], 0.0, 0.97)[None, :])
        fg2a = fga - fg3a
        rim = rng.binomial(fg2a, np.clip(g("prim"), 0.0, 1.0)[None, :])
        mid = fg2a - rim
        home_f = (self.r.home_eff if home else 1.0 / self.r.home_eff) * getattr(self.r, "env_eff", 1.0)

        def skill(pct, nn, dmult):
            a, b = pct * nn, (1 - pct) * nn
            draw = rng.beta(a[None, :].repeat(n, 0), b[None, :].repeat(n, 0))
            return np.clip(draw * dmult * home_f * eff[:, None], 0.01, 0.99)

        p3 = skill(g("pct3"), g("n3"), D["d_pct3"])
        prim = skill(g("pctrim"), g("nrim"), D["d_pctrim"])
        pmid = skill(g("pctmid"), g("nmid"), D["d_pctmid"])
        fg3m = rng.binomial(fg3a, p3)
        rimm = rng.binomial(rim, prim)
        midm = rng.binomial(mid, pmid)
        fgm = fg3m + rimm + midm
        extra = np.clip(g("fta_trip") - 1.0, 0.0, 1.0)
        fta = trips + rng.binomial(trips, np.broadcast_to(extra, trips.shape))
        ftm = rng.binomial(fta, np.clip(g("pctft"), 0.0, 1.0)[None, :])
        pts = 2 * (rimm + midm) + 3 * fg3m + ftm
        return {"players": t.available, "R": R, "T": T, "M": M, "form": form, "fga": fga, "fgm": fgm, "fg3a": fg3a, "fg3m": fg3m,
                "fta": fta, "ftm": ftm, "tov": tov, "pts": pts}

    def _rebounds_assists(self, rng, A: dict, B: dict) -> None:
        """Rebounds on A's misses (A offensive, B defensive); assists on A's makes."""
        n = A["fga"].shape[0]
        miss = (A["fga"] - A["fgm"]).sum(1)
        lgT = self.r.LT
        po = A["T"]["oreb_t"] * (1 - B["T"]["dreb_t"]) / (1 - lgT["dreb_t"])
        pd_ = B["T"]["dreb_t"] * (1 - A["T"]["oreb_t"]) / (1 - lgT["oreb_t"])
        s = po + pd_
        if s > 0.97:
            po, pd_ = po * 0.97 / s, pd_ * 0.97 / s
        out = rng.multinomial(miss, [po, pd_, 1 - po - pd_])
        orA = np.array([x["oreb"] for x in A["R"]])
        drB = np.array([x["dreb"] for x in B["R"]])
        A["oreb"] = _multinomial(rng, out[:, 0], orA[None, :] * A["M"] * A["form"])
        B["dreb"] = _multinomial(rng, out[:, 1], drB[None, :] * B["M"] * B["form"])
        team_fgm = A["fgm"].sum(1)
        assisted = rng.binomial(team_fgm, A["T"]["ast_t"])
        astA = np.array([x["ast"] for x in A["R"]])
        wa = astA[None, :] * A["M"] * A["form"]
        if self.j.ast_on_makes:
            others = np.maximum(team_fgm[:, None] - A["fgm"], 0)
            exp_others = np.maximum(others.mean(0, keepdims=True), 1e-9)
            wa = wa * others / exp_others
        ast = _multinomial(rng, assisted, wa)
        cap = team_fgm[:, None] - A["fgm"]                     # cannot assist your own basket
        clipped = np.minimum(ast, cap)
        A["assist_clip"] = float((ast - clipped).sum() / max(1, ast.sum()))
        A["ast"] = clipped
        A["miss"], A["team_fgm"] = miss, team_fgm

    # ------------------------------------------------------------------ public
    def _run(self, home: TeamInput, away: TeamInput, n: int, seed: int, latent: np.ndarray | None,
             mu: float = 0.0, rho: float = 0.0, exp_pts: float = 115.0):
        ss = np.random.SeedSequence([seed, 2026])
        r_min, r_pace, r_h, r_a, r_ra = (np.random.default_rng(s) for s in ss.spawn(5))
        if latent is None:
            bx = np.zeros(n)
            n_ot = np.zeros(n, dtype=int)
            shock = np.zeros(n)
        else:
            tau = self.m.sd_margin * norm.ppf(0.5 + self.m.p_ot / 2)
            tie = np.abs(latent) < tau
            n_ot = np.where(tie, r_min.geometric(1 - self.m.p_ot_cont, n), 0)
            bx = self.m.bx_from_margin(np.where(tie, 0.0, latent), n_ot, seed)
            shock = rho * (latent - mu) / (2.0 * exp_pts)
        mg = None if latent is None else np.where(n_ot > 0, 0.0, latent)
        mh = self.m.simulate_team(home.team_id, home.available, home.starters, home.caps, bx, n_ot, seed, margin=mg)
        ma = self.m.simulate_team(away.team_id, away.available, away.starters, away.caps, bx, n_ot, seed + 1,
                                  margin=None if mg is None else -mg)
        gm = 48.0 + 5.0 * n_ot
        pace = r_pace.normal(self.r.expected_pace(home.team_id, away.team_id), self.r.pace_sd, n)
        P = np.maximum(60, np.round(pace * gm / 48.0)).astype(int)
        H = self._team(r_h, home, away.team_id, mh.samples, P, gm, 1.0 + shock, True)
        A = self._team(r_a, away, home.team_id, ma.samples, P, gm, 1.0 - shock, False)
        self._rebounds_assists(r_ra, H, A)
        self._rebounds_assists(r_ra, A, H)
        return H, A, n_ot, P

    def rating_margin(self, home_id: int, away_id: int) -> float:
        m = self.m
        return m.ratings.get(home_id, 0.0) - m.ratings.get(away_id, 0.0) + m.home_edge

    def simulate(self, home: TeamInput, away: TeamInput, n: int = 10_000, seed: int = 0,
                 pilot_n: int = 2000, oversample: float = 1.6, anchor: str = "rating",
                 anchor_margin: float | None = None) -> GameSim:
        """Two passes:
        1. pilot (no script) -> bottom-up mean margin mu and expected points.
        2. main run: latent script L ~ N(mu, sd_margin_t) sets blowouts/OT for minutes and shifts
           team efficiency so the simulated margin tracks L (corr(L, final) high).
        3. calibration: the bottom-up process has no score effects, so its (margin, total) spread is
           wider than real games. Whole simulations are importance-resampled so that (margin, total)
           match the predictive SDs measured on real data, keeping the bottom-up means. Every kept
           simulation is an intact, accounting-consistent game.
        """
        H0, A0, _, _ = self._run(home, away, pilot_n, seed + 7, None)
        m0 = H0["pts"].sum(1) - A0["pts"].sum(1)
        mu = float(m0.mean())
        exp_pts = float((H0["pts"].sum(1).mean() + A0["pts"].sum(1).mean()) / 2)
        sd_m, sd_t = self.r.sd_margin_t, self.r.sd_total_t
        N = int(n * oversample)
        # anchor="rating": the script's expected margin comes from the team point-differential
        # rating model (validated better on real 2025-26 games); player-level stats then shift
        # consistently through the efficiency shock. anchor="bottom_up": pilot mean (V1.0).
        if anchor_margin is not None:          # external rating (e.g. NBA_TEAM_RATING_V2 blend)
            mu_script, anchor = float(anchor_margin), "external"
        else:
            mu_script = self.rating_margin(home.team_id, away.team_id) if anchor == "rating" else mu
        latent = np.random.default_rng([seed, 99]).normal(mu_script, sd_m, N)
        H, A, n_ot, P = self._run(home, away, N, seed, latent, mu, 1.0, exp_pts)
        hs, as_ = H["pts"].sum(1).astype(float), A["pts"].sum(1).astype(float)
        mg, tt = hs - as_, hs + as_
        # Gaussian fit of the simulated (margin, total), target = same means, measured SDs, independent
        C = np.cov(np.vstack([mg, tt]))
        mean = np.array([mu_script if anchor in ("rating", "external") else mg.mean(), tt.mean()])
        T = np.diag([sd_m ** 2, sd_t ** 2])
        z = np.vstack([mg, tt]).T - mean
        logw = -0.5 * np.einsum("ij,jk,ik->i", z, np.linalg.inv(T), z) + 0.5 * np.einsum("ij,jk,ik->i", z, np.linalg.inv(C), z)
        tilt_ok = np.all(np.diag(C) > np.diag(T))
        if tilt_ok:
            w = np.exp(logw - logw.max())
            w /= w.sum()
            ess = float(1.0 / np.sum(w ** 2))
            u = (np.random.default_rng([seed, 5]).random() + np.arange(n)) / n      # systematic resampling
            idx = np.minimum(np.searchsorted(np.cumsum(w), u), N - 1)
        else:   # simulated spread already narrower than reality: no tilt (flagged)
            ess, idx = float(n), np.arange(n)

        def pack(X, t):
            st = {s: X[s][idx] for s in ("pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "tov")}
            st["minutes"] = X["M"][idx]
            st["reb"] = st["oreb"] + st["dreb"]
            return TeamSim(t.team_id, list(t.available), st, st["pts"].sum(1), X["miss"][idx], X["team_fgm"][idx],
                           X["assist_clip"])
        meta = {"sim": self.NAME, "version": self.VERSION, "n": n, "seed": seed, "pilot_mu": mu,
                "anchor": anchor, "script_mu": mu_script,
                "raw_sd_margin": float(np.sqrt(C[0, 0])), "raw_sd_total": float(np.sqrt(C[1, 1])),
                "target_sd_margin": sd_m, "target_sd_total": sd_t, "ess": ess, "tilt_applied": bool(tilt_ok),
                "corr_latent_final": float(np.corrcoef(latent, mg)[0, 1]),
                "minutes_engine": self.m.fingerprint(), "rates": self.r.fingerprint(), "joint": self.j.tag(),
                "flags": list(self.r.score_flags) + ([] if tilt_ok else ["SIM_SPREAD_BELOW_TARGET"])}
        return GameSim(pack(H, home), pack(A, away), n_ot[idx], latent[idx], P[idx], meta)


def accounting_check(g: GameSim) -> dict[str, bool]:
    out = {}
    for name, T, O in (("home", g.home, g.away), ("away", g.away, g.home)):
        s = T.stats
        out[f"{name}_points"] = bool(np.array_equal(s["pts"].sum(1), T.score))
        out[f"{name}_points_from_events"] = bool(np.array_equal(
            s["pts"], 2 * (s["fgm"] - s["fg3m"]) + 3 * s["fg3m"] + s["ftm"]))
        out[f"{name}_rebounds"] = bool(np.all(s["oreb"].sum(1) + O.stats["dreb"].sum(1) <= T.misses))
        out[f"{name}_assists"] = bool(np.all(s["ast"] <= T.fgm[:, None] - s["fgm"]))
        out[f"{name}_makes"] = bool(np.all((s["fgm"] <= s["fga"]) & (s["fg3m"] <= s["fg3a"]) & (s["ftm"] <= s["fta"])))
        out[f"{name}_minutes"] = bool(np.allclose(s["minutes"].sum(1), 240 + 25 * g.n_ot))
    return out
