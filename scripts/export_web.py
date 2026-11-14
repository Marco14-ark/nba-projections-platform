"""Public API export: precomputed, read-only JSON for the web platform.

Reads ONLY stored results (prospective.db, shadow DBs, registry, archived logs). Never runs the model,
never alters a snapshot. Every number comes from a frozen pre-tip snapshot or an official result.
  python scripts/export_web.py OUT_DIR [--days 3]
Writes: meta.json, slate.json, games/<gid>.json, players/index.json, players/<pid>.json, results.json
Private fields are never exported (no account data, no file paths, no input payloads)."""
import json, os, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from nba_platform.core.timeutil import ET
from nba_platform.sim.calibration_v2 import apply_production
from nba_platform.sources.teams import TEAMS
from nba_platform.odds import evaluate as odds_eval
from nba_platform.injuries import impact as inj_impact
from nba_platform.odds.store import OddsStore

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
OUT = sys.argv[1]
DAYS = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 3
NOW = datetime.now(timezone.utc)
TEAM = {t[0]: {"id": t[0], "tri": t[1], "city": t[2], "name": t[3]} for t in TEAMS}
CAL = json.load(open(os.path.join(ROOT, "models/calibration_beta_mkt.json")))
REG = json.load(open(os.path.join(ROOT, "models/registry.json")))
PR = json.load(open(os.path.join(ROOT, "models/play_rates_hist.json")))
MARKETS = ("PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "FGA", "FGM", "3PA")
os.makedirs(os.path.join(OUT, "games"), exist_ok=True)
os.makedirs(os.path.join(OUT, "players"), exist_ok=True)


def r(x, k=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def db(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


c = db(os.path.join(D, "live", "prospective.db"))
names = {x["player_id"]: dict(x) for x in c.execute("SELECT player_id, full_name, first_name, last_name, position FROM players")}


def pname(pid):
    n = names.get(pid)
    return n["full_name"] if n and n["full_name"] else pid


def cond_text(cond):
    return [{"player_id": p, "player": pname(p), "state": s} for p, s in cond.items()]


# ------------------------------------------------------------------ games in scope
today_et = NOW.astimezone(ET).date()
dates = [(today_et + timedelta(days=i)).isoformat() for i in range(-1, DAYS)]
games = [dict(g) for g in c.execute(
    f"SELECT * FROM games WHERE game_date_et IN ({','.join('?' * len(dates))}) ORDER BY tip_utc", dates)]
slate, player_refs = [], defaultdict(list)
logs = pd.read_pickle(os.path.join(D, "real_logs_v3.pkl"))
season = logs[logs.game_date_et >= "2026-10-01"]
last_team = season.sort_values("game_date_et").groupby("player_id").team_id.last().to_dict()
integrity = {"team_mismatch": [], "post_tip_snapshot": 0, "coherence_max_gap": 0.0}
# ------------------------------------------------------------------ sportsbook odds (real captures only)
ODDS_DB = os.path.join(D, "odds", "odds.db")
ODDS_STATUS = json.load(open(os.path.join(D, "odds", "odds_status.json"))) if os.path.exists(os.path.join(D, "odds", "odds_status.json")) else None
GATE = json.load(open(os.path.join(D, "odds", "market_gate.json"))) if os.path.exists(os.path.join(D, "odds", "market_gate.json")) else {}
odds_store = OddsStore(ODDS_DB) if os.path.exists(ODDS_DB) else None
CAL_PARAMS = {"markets": {k: {x: v[x] for x in ("a", "b", "c")} for k, v in CAL["markets"].items()}, "eps": 1e-4}
opportunities = []
HAS_GDIST = bool(c.execute("SELECT 1 FROM sqlite_master WHERE name='game_distributions'").fetchone())

for g in games:
    snap = c.execute("""SELECT * FROM forecast_snapshots WHERE game_id=? AND created_at < ? ORDER BY created_at DESC LIMIT 1""",
                     (g["game_id"], g["tip_utc"])).fetchone()
    home, away = TEAM.get(g["home_team_id"]), TEAM.get(g["away_team_id"])
    if not home or not away:
        continue
    status = {1: "SCHEDULED", 2: "LIVE", 3: "FINAL"}.get(g["status"], "SCHEDULED")
    base = {"game_id": g["game_id"], "date_et": g["game_date_et"], "tip_utc": g["tip_utc"], "status": status,
            "home": home, "away": away,
            "final": {"home": g["home_score"], "away": g["away_score"]} if status == "FINAL" else None}
    if snap is None:
        slate.append({**base, "forecast": None})
        continue
    if snap["created_at"] >= g["tip_utc"]:
        integrity["post_tip_snapshot"] += 1
        continue
    sid = snap["snapshot_id"]
    ctx = json.loads(snap["context"])
    pubs = {(p["component"], p["team_id"]): dict(p) for p in c.execute("SELECT * FROM forecast_publications WHERE snapshot_id=?", (sid,))}
    for p in pubs.values():
        p["scenarios"] = json.loads(p["scenarios"])
    payload = json.loads(c.execute("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,)).fetchone()[0])
    gp = pubs.get(("GAME", 0))
    labels = [gp["published_label"]] if gp and gp["published_label"] else [k for k in payload if k != "MIX"]

    def mk(label):
        m = payload[label]["markets"]
        return {"proj_home": r(m["projected_home"], 1), "proj_away": r(m["projected_away"], 1),
                "p_home_win": r(m["p_home_win"], 4), "p_away_win": r(m["p_away_win"], 4),
                "fair_ml_home": r(m["fair_ml_home"], 0), "fair_ml_away": r(m["fair_ml_away"], 0),
                "fair_spread_home": r(m["model_fair_spread_home"], 1), "fair_total": r(m["model_fair_total"], 1),
                "p_ot": r(m["p_overtime"], 4),
                "margin": {k: r(v, 1) for k, v in m["margin"].items()}, "total": {k: r(v, 1) for k, v in m["total"].items()},
                "team_total_home": {k: r(v, 1) for k, v in m["team_total_home"].items()},
                "team_total_away": {k: r(v, 1) for k, v in m["team_total_away"].items()},
                "alt_spreads_home": [{"line": a["spread"], "p": r(a["p_cover"], 4)} for a in m["alt_spreads_home"]],
                "alt_totals": [{"line": a["total"], "p_over": r(a["p_over"], 4)} for a in m["alt_totals"]]}

    scen_info = {s["label"]: s for s in (gp["scenarios"] if gp else [])}
    game_markets = [{"label": lb, "conditions": cond_text(scen_info.get(lb, {}).get("conditions", {})),
                     "weight": scen_info.get(lb, {}).get("weight"), "markets": mk(lb)} for lb in labels]
    # injuries: official observations attached to this game, latest per player, with source and timestamps
    inj = defaultdict(list)
    for o in c.execute("""SELECT * FROM injury_status WHERE game_id=? ORDER BY observed_at""", (g["game_id"],)):
        inj[o["player_id"]] = dict(o)
    injuries = {"home": [], "away": []}
    for pid, o in inj.items():
        side = "home" if o["team_id"] == g["home_team_id"] else "away"
        injuries[side].append({"player_id": pid, "player": pname(pid), "status": o["status"], "reason": o["reason"],
                               "source": o["source"], "report_ts": o["report_ts"], "observed_at": o["observed_at"],
                               "after_snapshot": o["observed_at"] > snap["created_at"]})
    # props: published label(s) for the PROPS component
    pp = pubs.get(("PROPS", 0))
    plabels = [pp["published_label"]] if pp and pp["published_label"] else sorted({x[0] for x in c.execute(
        "SELECT DISTINCT scenario FROM prop_forecasts WHERE snapshot_id=?", (sid,))})
    pscen = {s["label"]: s for s in (pp["scenarios"] if pp else [])}
    rows = c.execute(f"""SELECT * FROM prop_forecasts WHERE snapshot_id=? AND scenario IN ({','.join('?' * len(plabels))})
                         AND market IN ({','.join('?' * len(MARKETS))})""", (sid, *plabels, *MARKETS)).fetchall()
    players = {}
    for p in rows:
        key = (p["scenario"], p["player_id"])
        if key not in players:
            tid = p["team_id"]
            opp = g["away_team_id"] if tid == g["home_team_id"] else g["home_team_id"]
            if last_team.get(p["player_id"]) not in (None, tid):
                integrity["team_mismatch"].append([p["player_id"], tid, last_team.get(p["player_id"])])
            players[key] = {"player_id": p["player_id"], "player": pname(p["player_id"]),
                            "pos": (names.get(p["player_id"]) or {}).get("position"), "team": TEAM[tid]["tri"],
                            "opp": TEAM[opp]["tri"], "home": tid == g["home_team_id"], "scenario": p["scenario"],
                            "minutes": {"mean": r(p["minutes_mean"], 1)}, "p_dnp": r(p["p_dnp"], 3), "markets": {}}
        q_raw = p["p_over_fair"]
        q_cal = float(apply_production(p["market"], [q_raw], CAL)[0]) if q_raw is not None and p["market"] in CAL["markets"] else None
        players[key]["markets"][p["market"]] = {
            "mean": r(p["mean"]), "median": r(p["median"], 1), "p10": r(p["p10"], 1), "p25": r(p["p25"], 1),
            "p75": r(p["p75"], 1), "p90": r(p["p90"], 1), "sd": r(p["sd"]), "fair_line": r(p["model_fair_line"], 1),
            "p_over_raw": r(q_raw, 4), "p_over": r(q_cal, 4), "p_under": r(1 - q_cal, 4) if q_cal is not None else None,
            "fair_odds_over": american(q_cal), "fair_odds_under": american(1 - q_cal) if q_cal is not None else None}
    # full Monte Carlo distributions (V6 snapshots onward); older snapshots have none -> calculator disabled
    for dr in c.execute(f"""SELECT * FROM prop_distributions WHERE snapshot_id=? AND scenario IN ({','.join('?' * len(plabels))})""",
                        (sid, *plabels)):
        key = (dr["scenario"], dr["player_id"])
        if key in players:
            players[key].setdefault("dist", {})[dr["market"]] = {"n": dr["n"], "lo": dr["lo"], "c": json.loads(dr["counts"])}
    # minutes distribution of the published minutes label per team
    mins = {}
    for side in ("home", "away"):
        tid = g[f"{side}_team_id"]
        mp = pubs.get(("MINUTES", tid))
        mlabels = [mp["published_label"]] if mp and mp["published_label"] else [s["label"] for s in (mp or {}).get("scenarios", [])]
        for m in c.execute(f"""SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND team_id=? AND scenario IN
                               ({','.join('?' * len(mlabels))})""", (sid, tid, *mlabels)):
            mins[(m["scenario"], m["player_id"])] = {"mean": r(m["mean"], 1), "p10": r(m["p10"], 1), "p90": r(m["p90"], 1),
                                                     "median": r(m["median"], 1)}
    for (lb, pid), pl in players.items():
        # props label -> team minutes label: MIX<->MIXTURE, H{i}A{j}<->S{i}/S{j}, BASE<->BASE
        cand = [v for (ml, mpid), v in mins.items() if mpid == pid]
        if len(cand) == 1:
            pl["minutes"].update(cand[0])
    late = {i["player_id"]: i for side in ("home", "away") for i in injuries[side]
            if i["after_snapshot"] and i["status"] in ("OUT", "DOUBTFUL", "NOT_WITH_TEAM", "SUSPENDED")}
    for pl in players.values():
        if pl["player_id"] in late and (pl["minutes"].get("mean") or 0) > 0:
            pl["late_status"] = {k: late[pl["player_id"]][k] for k in ("status", "reason", "report_ts")}
    starters = {}
    for side in ("home", "away"):
        sc = ctx["scenarios"][side][0]
        starters[side] = {"state": "CONFIRMED" if (ctx.get("confirmed_starters") or {}).get(side) else ctx[side]["lineup_state"],
                          "players": [{"player_id": s, "player": pname(s)} for s in sc["starters"]]}
    by_team = defaultdict(list)
    for pl in players.values():
        if pl["scenario"] == plabels[0]:
            by_team["home" if pl["home"] else "away"].append(pl)
    key_players = {s: [{"player_id": x["player_id"], "player": x["player"], "min": x["minutes"].get("mean"),
                        "pts": x["markets"].get("PTS", {}).get("mean"), "reb": x["markets"].get("REB", {}).get("mean"),
                        "ast": x["markets"].get("AST", {}).get("mean")}
                       for x in sorted(v, key=lambda z: -(z["minutes"].get("mean") or 0))[:3]] for s, v in by_team.items()}
    # V9: injury-impact-ranking-v1 (presentation triage only; the model already used every absence)
    important = {}
    lab0 = plabels[0] if plabels else None
    for side in ("home", "away"):
        tid = g[f"{side}_team_id"]
        proj = {pl["player_id"]: {"min": pl["minutes"].get("mean"), "pts": (pl["markets"].get("PTS") or {}).get("mean")}
                for pl in players.values() if pl["scenario"] == lab0 and pl["home"] == (side == "home")}
        absn = [i for i in injuries[side] if i["status"] != "AVAILABLE"]
        ranked = inj_impact.rank_absences(season, tid, g["game_date_et"], absn,
                                          {k: (v or {}).get("position") for k, v in names.items()}, proj, {})
        for o in ranked:
            for b in o["beneficiaries"]:
                b["player"] = pname(b["player_id"])
            # computed effect only where the model actually simulated both cases (conditional scenarios)
            eff = {"PLAYS": [], "OUT": []}
            for gm_ in game_markets:
                for cnd in gm_["conditions"]:
                    if cnd["player_id"] == o["player_id"]:
                        pw = gm_["markets"]["p_home_win"] if side == "home" else gm_["markets"]["p_away_win"]
                        eff["OUT" if cnd["state"] == "OUT" else "PLAYS"].append(pw)
            o["scenario_effect"] = ({"team_win_if_plays": r(np.mean(eff["PLAYS"]), 4), "team_win_if_out": r(np.mean(eff["OUT"]), 4)}
                                    if eff["PLAYS"] and eff["OUT"] else None)
            o.pop("score", None)
        important[side] = ranked
    forecast = {"snapshot_id": sid, "window": snap["refresh_window"], "created_at": snap["created_at"], "flag": snap["flag"],
                "model": snap["model_name"], "model_version": snap["model_version"], "stack": ctx.get("stack"),
                "calibration": ctx.get("calibration"), "input_hash": snap["input_hash"][:12],
                "minutes_before_tip": r(ctx.get("minutes_before_tip"), 0), "data_flags": ctx.get("data_flags", []),
                "injury_report": {k: (ctx.get("sources") or {}).get(k) for k in ("injury_report_url", "captured_at")},
                "publication": {"GAME": gp["publication"] if gp else None, "PROPS": pp["publication"] if pp else None,
                                "weights_source": (gp or {}).get("weights_source")},
                "game_markets": game_markets, "starters": starters, "injuries": injuries, "important_absences": important,
                "impact_model": inj_impact.VERSION, "key_players": key_players,
                "has_distributions": any("dist" in p for p in players.values()),
                "late_status_conflicts": [{"player": pname(p), **{k: v[k] for k in ("status", "report_ts")}} for p, v in late.items()]}
    for pl in players.values():
        player_refs[pl["player_id"]].append({"game_id": g["game_id"], "date_et": g["game_date_et"], "opp": pl["opp"],
                                             "home": pl["home"], "scenario": pl["scenario"], "minutes": pl["minutes"],
                                             "markets": {k: {x: v[x] for x in ("mean", "median", "p10", "p90")} for k, v in pl["markets"].items()}})
    # V8: exact game-level Monte Carlo distributions of the published label(s) (same simulations as props)
    gdist = defaultdict(dict)
    for dr in ([] if not HAS_GDIST else c.execute(f"""SELECT * FROM game_distributions WHERE snapshot_id=? AND scenario IN ({','.join('?' * len(labels))})""",
                        (sid, *labels))):
        gdist[dr["scenario"]][dr["market"]] = {"n": dr["n"], "lo": dr["lo"], "c": json.loads(dr["counts"])}
    for gm_ in game_markets:
        if gm_["label"] in gdist:
            gm_["dist"] = gdist[gm_["label"]]
    forecast["has_game_distributions"] = bool(gdist)
    # coherence: player points (same simulations) must add up to the team totals
    coh = {}
    for gm_ in game_markets:
        for side, key in (("home", "proj_home"), ("away", "proj_away")):
            sp = sum((pl["markets"].get("PTS") or {}).get("mean") or 0 for pl in players.values()
                     if pl["scenario"] == gm_["label"] and pl["home"] == (side == "home"))
            if any(pl["scenario"] == gm_["label"] for pl in players.values()):
                coh.setdefault(gm_["label"], {})[side] = {"team": gm_["markets"][key], "sum_players": r(sp, 1)}
                integrity["coherence_max_gap"] = max(integrity["coherence_max_gap"], abs((gm_["markets"][key] or 0) - sp))
    forecast["coherence"] = coh
    # sportsbook lines vs model (only real captured quotes; nothing when no provider is configured)
    odds_rows = []
    if odds_store is not None:
        qs = odds_store.current(game_id=g["game_id"])
        if qs:
            single = len(labels) == 1
            lb0 = labels[0]
            gd = {k: (v["lo"], v["c"]) for k, v in gdist.get(lb0, {}).items()} if single else None
            pdists = {pl["player_id"]: {m: (v["lo"], v["c"]) for m, v in (pl.get("dist") or {}).items()}
                      for pl in players.values() if pl["scenario"] == (plabels[0] if len(plabels) == 1 else None)}
            pregame = status == "SCHEDULED" and NOW < datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00"))
            odds_rows = odds_eval.build(qs, gd, pdists, CAL_PARAMS, GATE, NOW, pregame,
                                        set(late) | {i["player_id"] for s_ in ("home", "away") for i in injuries[s_]
                                                     if i["after_snapshot"] and i["status"] != "AVAILABLE"}, not single or len(plabels) != 1,
                                        {pid: pname(pid) for pid in {q.get("player_id") for q in qs} if pid})
            for o in odds_rows:
                if o.get("ev") is not None:
                    opportunities.append({"game_id": g["game_id"], "game": f"{away['tri']} @ {home['tri']}",
                                          "tip_utc": g["tip_utc"], **o})
    forecast["odds"] = {"status": "OK" if odds_rows else "NO_MARKET_DATA", "rows": odds_rows,
                        "books": sorted({o["book"] for o in odds_rows})}
    detail = {**base, "forecast": forecast,
              "props": {"labels": [{"label": lb, "conditions": cond_text(pscen.get(lb, {}).get("conditions", {})),
                                    "weight": pscen.get(lb, {}).get("weight")} for lb in plabels],
                        "players": sorted(players.values(), key=lambda x: (x["scenario"], not x["home"], -(x["minutes"].get("mean") or 0)))}}
    json.dump(detail, open(os.path.join(OUT, "games", f"{g['game_id']}.json"), "w"), separators=(",", ":"))
    lite = {k: forecast[k] for k in ("snapshot_id", "window", "created_at", "flag", "publication", "starters", "key_players",
                                     "minutes_before_tip", "has_distributions", "late_status_conflicts")}
    lite["injury_count"] = {s: len(v) for s, v in injuries.items()}
    lite["has_game_distributions"] = forecast["has_game_distributions"]
    lite["odds_status"] = forecast["odds"]["status"]
    lite["odds_books"] = forecast["odds"]["books"]
    lite["injuries"] = {s: [i for i in v if i["status"] not in ("AVAILABLE",)] for s, v in injuries.items()}
    lite["important_absences"] = {s_: [{k: o[k] for k in ("player_id", "player", "status", "category", "long_absence")}
                                       for o in v if o["show_main"]] for s_, v in important.items()}
    lite["markets"] = game_markets[0]["markets"] if len(game_markets) == 1 else None
    lite["scenarios"] = [{"label": x["label"], "conditions": x["conditions"], "weight": x["weight"],
                          "p_home_win": x["markets"]["p_home_win"], "proj_home": x["markets"]["proj_home"],
                          "proj_away": x["markets"]["proj_away"], "total": x["markets"]["total"]["mean"]} for x in game_markets]
    slate.append({**base, "forecast": lite})

# ------------------------------------------------------------------ players
pl_index = []
by_team_file = defaultdict(dict)
season_played = season[season.played == 1]
for pid in sorted(player_refs):
    cur_tid = next((g_["home_team_id"] if u["home"] else g_["away_team_id"] for u in player_refs[pid]
                    for g_ in [next(x for x in games if x["game_id"] == u["game_id"])]), None)
    s_all = season_played[season_played.player_id == pid].sort_values("game_date_et")
    s = s_all[s_all.team_id == cur_tid]          # never present another team's stats as current
    other = s_all[s_all.team_id != cur_tid]
    last10 = [{"date": x.game_date_et, "game_id": x.game_id, "min": r(x.minutes, 1), "pts": int(x.pts), "reb": int(x.reb),
               "ast": int(x.ast), "fg3m": int(x.fg3m), "fga": int(x.fga), "fgm": int(x.fgm), "fg3a": int(x.fg3a),
               "started": int(x.started)} for x in s.tail(10).itertuples()]
    avg = {k: r(s[k].mean(), 1) for k in ("minutes", "pts", "reb", "ast", "fg3m", "fga", "fgm", "fg3a")} if len(s) else {}
    hist = [dict(x) for x in c.execute("""SELECT game_id, market, settled_label, pred_mean, p10, p90, actual, abs_error, in_p10_p90
                                         FROM settlements WHERE player_id=? AND category='PROSPECTIVE' ORDER BY game_id""", (pid,))]
    team = player_refs[pid][0]
    acc = {}
    for h in hist:
        a_ = acc.setdefault(h["market"], {"n": 0, "abs_err": 0.0, "in": 0})
        a_["n"] += 1; a_["abs_err"] += h["abs_error"]; a_["in"] += int(h["in_p10_p90"] or 0)
    acc = {k: {"n": v["n"], "mae": r(v["abs_err"] / v["n"], 2), "cov80": r(v["in"] / v["n"], 3)} for k, v in acc.items()}
    info = {"player_id": pid, "player": pname(pid), "pos": (names.get(pid) or {}).get("position"),
            "team": TEAM.get(cur_tid, {}).get("tri"), "games_2026_27": int(len(s)), "season_avg": avg,
            "games_other_team_2026_27": int(len(other)), "accuracy": acc,
            "last10": last10, "upcoming": player_refs[pid], "prospective_results": hist}
    by_team_file[info["team"] or "FA"][pid] = info
    pl_index.append({"player_id": pid, "player": info["player"], "team": info["team"] or "FA", "pos": info["pos"]})
for tri, d in by_team_file.items():      # one file per team keeps the published file count small
    json.dump(d, open(os.path.join(OUT, "players", f"{tri}.json"), "w"), separators=(",", ":"))
json.dump(pl_index, open(os.path.join(OUT, "players", "index.json"), "w"), separators=(",", ":"))

# ------------------------------------------------------------------ results (categories never mixed)
def settled(cn, cat):
    S = pd.read_sql("SELECT * FROM settlements WHERE category=?", cn, params=(cat,))
    if not len(S):
        return {"games": 0}
    P = S[S.component != "GAME"]
    by = P.groupby("market").agg(n=("abs_error", "size"), mae=("abs_error", "mean"), cov80=("in_p10_p90", "mean")).round(3)
    G = S[S.component == "GAME"].pivot_table(index="game_id", columns="market", values=["pred_mean", "p10", "p90", "actual"], aggfunc="first")
    games_ = []
    for gid in G.index:
        gg = c.execute("SELECT * FROM games WHERE game_id=?", (gid,)).fetchone()
        games_.append({"game_id": gid, "date_et": gg["game_date_et"], "home": TEAM[gg["home_team_id"]]["tri"],
                       "away": TEAM[gg["away_team_id"]]["tri"], "final_home": gg["home_score"], "final_away": gg["away_score"],
                       "pred_margin": r(G.loc[gid, ("pred_mean", "MARGIN")], 1), "margin_p10": r(G.loc[gid, ("p10", "MARGIN")], 0),
                       "margin_p90": r(G.loc[gid, ("p90", "MARGIN")], 0), "pred_total": r(G.loc[gid, ("pred_mean", "TOTAL")], 1),
                       "total_p10": r(G.loc[gid, ("p10", "TOTAL")], 0), "total_p90": r(G.loc[gid, ("p90", "TOTAL")], 0),
                       "p_home_win": r(G.loc[gid, ("pred_mean", "HOME_WIN")], 3), "home_won": bool(G.loc[gid, ("actual", "HOME_WIN")])})
    return {"games": int(S.game_id.nunique()), "by_market": by.reset_index().to_dict("records"), "game_results": games_,
            "settle_methods": S.groupby("settle_method").size().to_dict()}

results = {"prospective": settled(c, "PROSPECTIVE"), "shadow": {}}
for f in sorted(os.listdir(os.path.join(D, "live"))):
    if f.startswith("shadow_") and f.endswith(".db"):
        results["shadow"][f[7:-3]] = settled(db(os.path.join(D, "live", f)), "SHADOW")
champ = [x for x in REG["STACK"] if x.get("role") == "CHAMPION"][-1]
results["historical"] = {
    "champion": champ["name"],
    "calibration": [x for x in REG["NBA_PROB_CALIBRATION_V2"] if x.get("role") == "PUBLISHED_LAYER"][-1]["walk_forward"],
    "minutes_blowout_challenger": [x for x in REG["NBA_MINUTES_ENGINE_V1"] if "BLOWOUT_GD" in str(x.get("version"))][-1]["walk_forward"],
    "challengers": [{"name": x["name"], "role": x["role"], "since": x.get("since")} for x in REG["STACK"] if x.get("role") == "SHADOW"],
    "play_rates": {"fitted_through": PR["fitted_through"], "rates": PR["rates"], "validation_2026_27": PR["validation_2026_27"]}}
GV = os.path.join(ROOT, "docs", "validation", "game_markets_v8.json")
if os.path.exists(GV):
    gv = json.load(open(GV))
    results["historical"]["game_markets_v8"] = {k: {x: v[x] for x in ("n_games", "dates", "moneyline", "spread", "total", "team_totals", "cohorts")}
                                                for k, v in gv.items() if k.split("|")[0] in ("STACK_V2_ENV", "STACK_V3_SIGMA", "STACK_V4_BLOWOUT")
                                                and k.split("|")[1] == "v3"}
V9R = os.path.join(ROOT, "docs", "validation", "v9_research.json")
if os.path.exists(V9R):
    v9 = json.load(open(V9R))
    results["historical"]["v9_research"] = {k: v9[k] for k in ("minutes_audit", "props_error_decomposition", "totals_decomposition",
                                                               "game_challengers", "injury_context", "team_totals", "correlations",
                                                               "minutes_challenger_profile_recal") if k in v9}
GR = os.path.join(D, "odds", "market_gate_report.json")
results["market_gate"] = json.load(open(GR)) if os.path.exists(GR) else None
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), separators=(",", ":"), default=str)

live = {}
lp = os.path.join(D, "live", "live_scores.json")
if os.path.exists(lp):
    live = json.load(open(lp))
json.dump(live, open(os.path.join(OUT, "live.json"), "w"), separators=(",", ":"))
last_final = c.execute("SELECT MAX(game_date_et) FROM games WHERE status=3").fetchone()[0]
inj_last = c.execute("SELECT MAX(report_ts), MAX(observed_at) FROM injury_status WHERE origin='OFFICIAL'").fetchone()
meta = {"generated_at": NOW.isoformat(timespec="seconds"), "champion": champ["name"],
        "calibration": CAL.get("version"), "results_through": last_final,
        "latest_injury_report": {"report_ts": inj_last[0], "captured_at": inj_last[1]},
        "snapshots_total": c.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0],
        "sportsbook_lines": {"status": ("OK" if opportunities else "NO_MARKET_DATA"),
                             "providers": (ODDS_STATUS or {}).get("providers"), "checked_at": (ODDS_STATUS or {}).get("generated_at"),
                             "market_gate": {k: v.get("status") for k, v in GATE.items()} or "NOT_VALIDATED_VS_MARKET"},
        "calibration_params": {"version": CAL.get("version"), "markets": {k: {x: v[x] for x in ("a", "b", "c")} for k, v in CAL["markets"].items()},
                               "eps": 1e-4, "valid_for": "half-point lines; fitted on lines within the model fair line +-1..6"}, "integrity": {"team_mismatch": len(integrity["team_mismatch"]),
                                                 "post_tip_snapshots_exported": integrity["post_tip_snapshot"]},
        "dates": dates}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
rank = {"VALUE": 0, "NO_BET": 1}
opportunities.sort(key=lambda o: (rank.get(o["decision"], 2), -(o.get("ev_conservative") or -9), o.get("odds_age_min") or 9e9))
json.dump({"generated_at": meta["generated_at"], "status": meta["sportsbook_lines"]["status"],
           "rows": opportunities[:600], "n_total": len(opportunities)},
          open(os.path.join(OUT, "opportunities.json"), "w"), separators=(",", ":"))
json.dump({"generated_at": meta["generated_at"], "games": slate}, open(os.path.join(OUT, "slate.json"), "w"), separators=(",", ":"))
print(json.dumps({**meta, "games": len(slate), "players": len(pl_index), "team_mismatch_list": integrity["team_mismatch"][:10]}, indent=1))
