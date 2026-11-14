"""NBA_DAILY_AUTOPILOT_V1 — slate discovery, refresh windows, dependency-aware recompute.

`tick(now)` is meant to be called by a cloud scheduler every ~5 minutes. It is cheap when
nothing changed: it rebuilds the input hash tree for each game in a due window and only runs
the minutes simulation if a layer upstream of MINUTES changed. Forecasts are immutable
snapshots; a newer one supersedes the older, which is kept for the track record.
"""
from __future__ import annotations

import json

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from ..core.enums import AvailabilityStatus as S, ForecastFlag, RefreshWindow as W
from ..core.hashing import HashTree, recompute_plan, sha
from ..core.timeutil import parse_utc, utcnow
from ..db.store import Store
from ..injuries.monitor import InjuryMonitor
from ..lineups.resolve import TeamContext, build_scenarios, expected_starters, resolve_team
from ..sim.game_sim import GameSimulator, TeamInput, accounting_check
from ..sim.markets import game_markets, player_props
from ..sim.rates import OpportunityRates
from ..minutes.engine import MinutesConfig, MinutesEngine
from ..minutes.forecast import forecast_game

AUTOPILOT_NAME = "NBA_DAILY_AUTOPILOT_V1"
OFFSETS_MIN = {W.T_6H: 360, W.T_90M: 90, W.T_30M: 30, W.T_15M: 15}


def due_window(tip_utc: datetime, now: datetime, done: set[str], use_15m: bool = True) -> W | None:
    """Latest window whose time has arrived and that has not run. Missed earlier windows are
    skipped (running them late adds nothing). No refreshes at/after tip."""
    if now >= tip_utc:
        return None
    if W.INITIAL.value not in done:
        return W.INITIAL
    latest = None
    for w, off in OFFSETS_MIN.items():
        if w is W.T_15M and not use_15m:
            continue
        if now >= tip_utc - timedelta(minutes=off):
            latest = w
    if latest is not None and latest.value not in done:
        return latest
    return None


def load_frames(store: Store, before_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = pd.DataFrame([dict(r) for r in store.logs_before(before_date)])
    games = pd.DataFrame([dict(r) for r in store.q(
        "SELECT * FROM games WHERE game_date_et < ? AND status=3", (before_date,))])
    if not len(logs):
        logs = pd.DataFrame(columns=["game_id", "player_id", "team_id", "game_date_et", "started", "played",
                                     "minutes", "not_playing_reason"])
    if not len(games):
        games = pd.DataFrame(columns=["game_id", "game_date_et", "home_team_id", "away_team_id",
                                      "home_score", "away_score", "periods"])
    return logs, games


@dataclass
class RefreshResult:
    game_id: str
    window: str
    status: str            # COMPUTED | SKIPPED_NO_CHANGE | FAILED
    snapshot_id: str | None
    plan: list[str]
    detail: str = ""


class Autopilot:
    NAME = AUTOPILOT_NAME

    def __init__(self, store: Store, cfg: MinutesConfig | None = None, n_sims: int = 10_000,
                 engine_factory: Callable[[str], MinutesEngine] | None = None,
                 play_rates_fn: Callable[[str], dict] | None = None, with_sim: bool = False,
                 sim_n: int = 10_000, anchor_margin_fn: Callable | None = None,
                 context_extra_fn: Callable | None = None, stack_name: str = "STACK_V2_ENV",
                 game_features: "pd.DataFrame | None" = None, rates_cfg=None):
        self.s, self.cfg, self.n_sims = store, cfg or MinutesConfig(), n_sims
        self._engines: dict[str, MinutesEngine] = {}
        self._rates: dict[str, OpportunityRates] = {}
        self.rates_cfg = rates_cfg               # None -> RatesConfig() (champion)
        self.with_sim, self.sim_n = with_sim, sim_n
        # anchor_margin_fn(game_row, home_inputs, away_inputs) -> expected home margin from
        # NBA_TEAM_RATING_V2 (0.295*point-diff + 0.845*lineup RAPM). None -> point-diff rating.
        self.anchor_margin_fn = anchor_margin_fn
        self.context_extra_fn = context_extra_fn
        self.stack_name = stack_name
        self.game_features = game_features      # e.g. PBP garbage time per finished game (blowout challenger)
        self.engine_factory = engine_factory or self._default_engine
        self.play_rates_fn = play_rates_fn
        self.play_rates_source = getattr(play_rates_fn, "source", "unspecified")

    def _default_engine(self, game_date: str) -> MinutesEngine:
        if game_date not in self._engines:
            logs, games = load_frames(self.s, game_date)
            if self.game_features is not None and len(games):
                games = games.merge(self.game_features, on="game_id", how="left")
            pos = {r["player_id"]: r["position"] for r in self.s.q("SELECT player_id, position FROM players")}
            self._engines[game_date] = MinutesEngine(logs, games, game_date, self.cfg, pos)
        return self._engines[game_date]

    def rates_for(self, game_date: str) -> OpportunityRates:
        if game_date not in self._rates:
            logs, games = load_frames(self.s, game_date)
            self._rates[game_date] = OpportunityRates(logs, games, game_date, self.rates_cfg)
        return self._rates[game_date]

    MAX_PAIRS_CONDITIONAL = 4        # conditional (unweighted) scenario pairs stored per game
    MAX_PAIRS_MIXTURE = 16           # weighted pairs simulated for a published mixture

    def _store_sim(self, snap_id: str, g, eng: MinutesEngine, home: TeamContext, away: TeamContext,
                   seed: int, play_rates: dict | None = None, created_at: str = "") -> dict:
        """Joint Monte Carlo per availability-scenario pair.

        Weighted scenarios (justified play rates) -> every pair is simulated and the published
        distribution is the exact draw-level mixture ('MIX'). Unweighted -> each pair is a
        separate CONDITIONAL forecast; no mixture is published. Too many pairs -> only the
        all-available pair is stored (CONDITIONAL, settled only if that condition happens)."""
        from ..sim.mixture import mix_game_sims
        from . import publication as pubm
        sim = GameSimulator(eng, self.rates_for(g["game_date_et"]))
        hs = build_scenarios(home, eng.rot_players.get(home.team_id, {}), play_rates)
        as_ = build_scenarios(away, eng.rot_players.get(away.team_id, {}), play_rates)
        pairs = [(i, j) for i in range(len(hs)) for j in range(len(as_))]
        weighted = all(sc.prob is not None for sc in hs + as_)
        limit = self.MAX_PAIRS_MIXTURE if weighted else self.MAX_PAIRS_CONDITIONAL
        if len(pairs) > limit:
            pairs = [(0, 0)]
            weighted = False
        meta, sims, wts = {}, [], []
        for i, j in pairs:
            label = "BASE" if len(hs) == 1 and len(as_) == 1 else f"H{i}A{j}"
            ti = []
            for ctx, sc in ((home, hs[i]), (away, as_[j])):
                av = ctx.available(sc)
                ti.append(TeamInput(ctx.team_id, av, expected_starters(av, eng.start_scores(ctx.team_id, av),
                                                                        ctx.confirmed_starters), ctx.minutes_caps))
            am = self.anchor_margin_fn(g, ti[0], ti[1]) if self.anchor_margin_fn else None
            gs = sim.simulate(ti[0], ti[1], n=self.sim_n, seed=seed, anchor_margin=am)
            if not all(accounting_check(gs).values()):
                raise RuntimeError("team accounting failed")
            self._insert_props(snap_id, gs, label)
            gm = game_markets(gs)
            meta[label] = {"markets": gm, "sim_meta": gs.meta,
                           "scenario": {"home": hs[i].name, "home_prob": hs[i].prob if weighted else None,
                                        "away": as_[j].name, "away_prob": as_[j].prob if weighted else None}}
            if weighted and len(pairs) > 1:
                sims.append(gs)
                wts.append(hs[i].prob * as_[j].prob)
        if sims:
            tot = sum(wts)
            mix, _ = mix_game_sims(sims, [w / tot for w in wts], seed)
            if not all(accounting_check(mix).values()):
                raise RuntimeError("mixture accounting failed")
            self._insert_props(snap_id, mix, "MIX")
            meta["MIX"] = {"markets": game_markets(mix), "sim_meta": mix.meta,
                           "scenario": {"mixture_of": list(meta), "weights": [w / tot for w in wts]}}
        self.s.x("INSERT INTO game_market_forecasts VALUES (?,?)", (snap_id, json.dumps(meta, default=float)))
        scen = pubm.pair_scenarios(meta)
        src = f"play_rates:{getattr(self, '_rates_source', self.play_rates_source)}" if sims else None
        for comp in ("PROPS", "GAME"):
            pubm.record(self.s, snap_id, comp, 0, scen, "MIX" if sims else None, src, created_at)
        return meta

    DIST_MARKETS = ("PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "FGA", "FGM", "3PA")

    def _insert_game_dists(self, snap_id: str, gs, label: str) -> None:
        h = np.asarray(gs.home.score, dtype=np.int64)
        a = np.asarray(gs.away.score, dtype=np.int64)
        for mk, x in (("MARGIN", h - a), ("TOTAL", h + a), ("HOME_PTS", h), ("AWAY_PTS", a)):
            lo = int(x.min())
            self.s.x("INSERT INTO game_distributions VALUES (?,?,?,?,?,?)",
                     (snap_id, label, mk, int(len(x)), lo,
                      json.dumps(np.bincount(x - lo).tolist(), separators=(",", ":"))))

    def _insert_props(self, snap_id: str, gs, label: str) -> None:
        from ..sim.markets import PROP_MARKETS
        self._insert_game_dists(snap_id, gs, label)
        for T in (gs.home, gs.away):
            for k, pid in enumerate(T.players):
                for mk in self.DIST_MARKETS:
                    x = sum(T.stats[s][:, k] for s in PROP_MARKETS[mk]).astype(np.int64)
                    lo = int(x.min())
                    counts = np.bincount(x - lo)
                    self.s.x("INSERT INTO prop_distributions VALUES (?,?,?,?,?,?,?)",
                             (snap_id, pid, label, mk, int(len(x)), lo, json.dumps(counts.tolist(), separators=(",", ":"))))
        for p in player_props(gs, markets=("PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA",
                                           "FGA", "FGM", "3PA", "FTA")):
            self.s.x("""INSERT INTO prop_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (snap_id, p["player_id"], p["team_id"], p["market"], label, p["mean"], p["median"], p["p10"],
                      p["p25"], p["p75"], p["p90"], p["sd"], p["minutes_mean"], p["p_dnp"], p["model_fair_line"],
                      p["fair_p_over"], p["fair_p_under"]))

    # ------------------------------------------------------------------ inputs & hashes
    def inputs(self, g, as_of: str) -> tuple[TeamContext, TeamContext, HashTree, MinutesEngine]:
        eng = self.engine_factory(g["game_date_et"])
        home = resolve_team(self.s, g["game_id"], g["home_team_id"], g["game_date_et"], as_of)
        away = resolve_team(self.s, g["game_id"], g["away_team_id"], g["game_date_et"], as_of)
        ht = HashTree()
        ht.set_input("ROSTER", {"h": home.roster, "a": away.roster})
        ht.set_input("INJURY", {"h": {p: s.value for p, s in home.statuses.items()},
                                "a": {p: s.value for p, s in away.statuses.items()},
                                "caps": [home.minutes_caps, away.minutes_caps],
                                "play_rates": self.play_rates_fn(g["game_date_et"]) if self.play_rates_fn else None})
        ht.set_input("STARTERS", {"h": [home.lineup_state.value, home.confirmed_starters],
                                  "a": [away.lineup_state.value, away.confirmed_starters]})
        ht.set_input("MODEL", {"engine": eng.fingerprint(), "n_sims": self.n_sims,
                               "sim": [self.rates_for(g["game_date_et"]).fingerprint(), self.sim_n]
                               if self.with_sim else None})
        return home, away, ht, eng

    def _coverage_flag(self, game_id: str, team_ids: tuple[int, int], as_of: str) -> list[str]:
        flags = []
        for t in team_ids:
            r = self.s.q("""SELECT submitted FROM injury_report_coverage WHERE game_id=? AND team_id=?
                            AND observed_at<=? ORDER BY observed_at DESC LIMIT 1""", (game_id, t, as_of))
            if not r:
                flags.append(f"NO_OFFICIAL_INJURY_REPORT:{t}")
            elif not r[0]["submitted"]:
                flags.append(f"INJURY_REPORT_NOT_SUBMITTED:{t}")
        return flags

    # ------------------------------------------------------------------ refresh
    def refresh_game(self, g, window: W, now: datetime) -> RefreshResult:
        as_of = now.isoformat()
        try:
            home, away, ht, eng = self.inputs(g, as_of)
            new_hashes = ht.resolved()
            prev = self.s.latest_forecast(g["game_id"])
            old_hashes = json.loads(prev["hashes"]) if prev else None
            plan = recompute_plan(old_hashes, new_hashes)
            if prev is not None and "MINUTES" not in plan:
                return RefreshResult(g["game_id"], window.value, "SKIPPED_NO_CHANGE", prev["snapshot_id"], plan)
            input_hash = new_hashes["MINUTES"]
            seed = int(sha({"g": g["game_id"], "h": input_hash}), 16) % (2 ** 31)
            rates = self.play_rates_fn(g["game_date_et"]) if self.play_rates_fn else None
            mbt = (parse_utc(g["tip_utc"]) - now).total_seconds() / 60
            if rates and mbt > (rates.get("_meta") or {}).get("max_minutes_before_tip", float("inf")):
                rates = None          # rates measured closer to tip than this window: not applicable -> CONDITIONAL
            self._rates_source = ((rates or {}).get("_meta") or {}).get("source", self.play_rates_source)
            fc = forecast_game(eng, home, away, g["home_team_id"], g["away_team_id"], self.n_sims, seed, rates)
            data_flags = self._coverage_flag(g["game_id"], (g["home_team_id"], g["away_team_id"]), as_of)
            data_flags += home.flags + away.flags + fc["margin_flags"] + fc["global_flags"]
            flag = ForecastFlag.PARTIAL_DATA if data_flags else ForecastFlag.OK
            if not home.roster or not away.roster:
                flag = ForecastFlag.UNAVAILABLE
            snap_id = f"{g['game_id']}:{window.value}:{input_hash}:{now.strftime('%Y%m%dT%H%M%S')}"
            context = {"window": window.value, "stack": self.stack_name, "data_flags": data_flags, "p_ot": fc["p_ot"],
                       "tip_utc": g["tip_utc"], "minutes_before_tip": round((parse_utc(g["tip_utc"]) - now).total_seconds() / 60, 1),
                       "available": {"home": home.available(), "away": away.available()},
                       "confirmed_starters": {"home": home.confirmed_starters, "away": away.confirmed_starters},
                       "p_blowout": fc["p_blowout"],
                       "home": {k: fc["home"][k] for k in ("team_id", "lineup_state", "out", "uncertain", "flags")},
                       "away": {k: fc["away"][k] for k in ("team_id", "lineup_state", "out", "uncertain", "flags")},
                       "statuses": {"home": {p: s.value for p, s in home.statuses.items()},
                                    "away": {p: s.value for p, s in away.statuses.items()}},
                       "scenarios": {side: [{"name": sc["name"], "prob": sc["prob"], "starters": sc["starters"]}
                                            for sc in fc[side]["scenarios"]] for side in ("home", "away")}}
            if self.context_extra_fn:
                context.update(self.context_extra_fn(g))
            if parse_utc(g["tip_utc"]) <= now:
                raise RuntimeError("refusing to create a forecast at/after tip-off")
            rows = []
            for side in ("home", "away"):
                t = fc[side]
                for i, sc in enumerate(t["scenarios"]):
                    for pid, m in sc["players"].items():
                        rows.append({**m, "player_id": pid, "team_id": t["team_id"],
                                     "scenario": "BASE" if len(t["scenarios"]) == 1 else f"S{i}"})
                if t["mixture"]:
                    for pid, m in t["mixture"].items():
                        rows.append({**m, "player_id": pid, "team_id": t["team_id"], "scenario": "MIXTURE"})
            self.s.insert_forecast({"snapshot_id": snap_id, "game_id": g["game_id"], "refresh_window": window.value,
                                    "created_at": as_of, "model_name": eng.NAME, "model_version": eng.VERSION,
                                    "input_hash": input_hash, "hashes": new_hashes, "flag": flag.value,
                                    "context": context, "seed": seed, "n_sims": self.n_sims}, rows)
            from . import publication as pubm
            for side in ("home", "away"):
                t = fc[side]
                scen = pubm.team_scenarios([{"name": sc["name"], "prob": sc["prob"]} for sc in t["scenarios"]])
                pubm.record(self.s, snap_id, "MINUTES", t["team_id"], scen, "MIXTURE" if t["mixture"] else None,
                            f"play_rates:{self._rates_source}" if t["mixture"] else None, as_of)
            if self.with_sim:
                self._store_sim(snap_id, g, eng, home, away, seed, rates, as_of)
            if prev is not None:
                self.s.supersede(prev["snapshot_id"], snap_id)
            self.s.commit()
            return RefreshResult(g["game_id"], window.value, "COMPUTED", snap_id, plan)
        except Exception as e:  # noqa: BLE001 - a failed game must not stop the slate
            self.s.conn.rollback()
            return RefreshResult(g["game_id"], window.value, "FAILED", None, [], repr(e))

    def windows_done(self, game_id: str) -> set[str]:
        return {r["refresh_window"] for r in self.s.q(
            "SELECT DISTINCT refresh_window FROM forecast_snapshots WHERE game_id=?", (game_id,))} | {
            r["detail"] for r in self.s.q("SELECT detail FROM job_runs WHERE job=? AND status='SKIPPED'",
                                          (f"refresh:{game_id}",))}

    def tick(self, now: datetime | None = None, game_date_et: str | None = None) -> list[RefreshResult]:
        now = now or utcnow()
        from ..core.timeutil import game_date_et as gd
        dates = {game_date_et} if game_date_et else {gd(now).isoformat(), gd(now + timedelta(hours=12)).isoformat()}
        out = []
        for d in sorted(dates):
            for g in self.s.games_on(d):
                if g["status"] != 1:
                    continue
                w = due_window(parse_utc(g["tip_utc"]), now, self.windows_done(g["game_id"]))
                if w is None:
                    continue
                res = self.refresh_game(g, w, now)
                job_id = f"{g['game_id']}:{w.value}"
                self.s.log_job(job_id, f"refresh:{g['game_id']}",
                               {"COMPUTED": "OK", "SKIPPED_NO_CHANGE": "SKIPPED"}.get(res.status, "FAILED"),
                               w.value if res.status == "SKIPPED_NO_CHANGE" else res.detail)
                self.s.commit()
                out.append(res)
        return out

    # ------------------------------------------------------------------ public read path
    def public_status(self, game_id: str, now: datetime | None = None) -> dict:
        """What the public site may show. Never runs a simulation (spec §62)."""
        now = now or utcnow()
        snap = self.s.latest_forecast(game_id)
        if snap is None:
            return {"flag": ForecastFlag.UNAVAILABLE.value, "snapshot_id": None}
        g = self.s.q("SELECT * FROM games WHERE game_id=?", (game_id,))[0]
        home, away, ht, _ = self.inputs(g, now.isoformat())
        cur = ht.resolved()
        old = json.loads(snap["hashes"])
        if cur["MINUTES"] == old["MINUTES"]:
            return {"flag": snap["flag"], "snapshot_id": snap["snapshot_id"]}
        ctx = json.loads(snap["context"])
        then = {**ctx["statuses"]["home"], **ctx["statuses"]["away"]}
        nowst = {**{p: s.value for p, s in home.statuses.items()}, **{p: s.value for p, s in away.statuses.items()}}
        core = {r["player_id"] for r in self.s.q(
            "SELECT player_id FROM minutes_forecasts WHERE snapshot_id=? AND role_tier='CORE'", (snap["snapshot_id"],))}
        changes = InjuryMonitor.diff({p: S(v) for p, v in then.items()}, {p: S(v) for p, v in nowst.items()},
                                     game_id, core)
        if any(c.relevance == "HIGH" for c in changes) or cur["ROSTER"] != old["ROSTER"]:
            return {"flag": ForecastFlag.UNAVAILABLE.value, "snapshot_id": snap["snapshot_id"],
                    "reason": "material change since forecast; recompute pending"}
        return {"flag": ForecastFlag.STALE.value, "snapshot_id": snap["snapshot_id"]}
