"""Live / prospective forecasts with the CHAMPION stack (STACK_V2_ENV).

  python scripts/live_forecast.py BUNDLE.json DB.sqlite [--now ISO] [--replay]

BUNDLE = capture from NBA.com (schedule, official injury PDF, pregame boxscores) + the 2026-27
play-by-play/boxscore archive already ingested into data/real_*_v3.pkl.
Every forecast is an immutable snapshot created strictly before tip-off. If a source is missing
the snapshot is flagged (PARTIAL_DATA / NO_OFFICIAL_INJURY_REPORT) — nothing is invented.
"""
import argparse, base64, gzip, json, os, sys, time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.environ.get("NBA_CODE_DIR", "/home/claude/nba-platform"))
from nba_platform.core.timeutil import ET, game_date_et, parse_utc, season_for_date, utcnow
from nba_platform.db.store import Store
from nba_platform.injuries.monitor import InjuryMonitor
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.pipeline.autopilot import Autopilot, load_frames
from nba_platform.pipeline.ingest import ensure_teams
from nba_platform.rating.team_rating import LineupRating, PointDiffRating, RatingConfig, rest_days
from nba_platform.roster.sync import RosterEntry, RosterSync
from nba_platform.sources.injury_report import parse_pdf

import os
from nba_platform.sim.rates import RatesConfig
D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
ap = argparse.ArgumentParser()
ap.add_argument("bundle"); ap.add_argument("db"); ap.add_argument("--now"); ap.add_argument("--replay", action="store_true")
ap.add_argument("--dates", nargs="*"); ap.add_argument("--sims", type=int, default=10_000)
ap.add_argument("--horizon-days", type=int, default=3)
ap.add_argument("--stack", default="STACK_V2_ENV"); ap.add_argument("--minutes-cfg", default="{}")
ap.add_argument("--rates-cfg", default="{}")
a = ap.parse_args()
now = parse_utc(a.now) if a.now else utcnow()
B = json.load(open(a.bundle))
s = Store(a.db)
ensure_teams(s)
t0 = time.time()

# 1) history (finished games only, strictly before `now`) — incremental: only games not yet in the store
G = pd.read_pickle(f"{D}/real_games_v3.pkl"); L = pd.read_pickle(f"{D}/real_logs_v3.pkl")
G = G[pd.to_datetime(G.tip_utc, utc=True, format="ISO8601") < pd.Timestamp(now)]
have = {r[0] for r in s.q("SELECT DISTINCT game_id FROM player_game_logs")}
newg = G[~G.game_id.isin(have)]
for r in newg.to_dict("records"): s.upsert_game(r)
for r in L[L.game_id.isin(set(newg.game_id))].to_dict("records"): s.upsert_log(r)
s.commit()
# 2) players with full names + effective-dated rosters from each team's latest official boxscore
latest = {}
with gzip.open(f"{D}/s2627/nba_2026_27_pbp_box.jsonl.gz", "rt") as f:
    for line in f:
        g = json.loads(line)["box"]
        tip = parse_utc(g["gameTimeUTC"])
        if tip >= now:
            continue
        for side in ("homeTeam", "awayTeam"):
            t = g[side]
            if t["teamId"] not in latest or latest[t["teamId"]][0] < tip:
                latest[t["teamId"]] = (tip, t["players"])
rs = RosterSync(s)
for tid, (tip, players) in latest.items():
    ents = [RosterEntry(int(tid), p.get("firstName", ""), p.get("familyName", ""), int(p["personId"]), position=p.get("position"))
            for p in players]
    rs.ingest_snapshot("nba_box_latest", game_date_et(tip).isoformat(), ents, covered_team_ids={int(tid)})
# 3) scheduled games
for g in B["schedule"]:
    if not g["h"] or not g["a"]:
        continue                      # knockout games with undetermined teams
    tip = parse_utc(g["t"]); d = game_date_et(tip)
    if g["st"] == 3 and s.q("SELECT 1 FROM games WHERE game_id=? AND status=3", (g["gameId"],)):
        continue
    s.upsert_game({"game_id": g["gameId"], "season": season_for_date(d), "game_date_et": d.isoformat(),
                   "tip_utc": tip.isoformat(), "home_team_id": int(g["h"]), "away_team_id": int(g["a"]), "status": int(g["st"])})
s.commit()
# 4) official injury report (as published; observed when captured)
inj = {"injury_report_url": B.get("injury_pdf_url"), "captured_at": B["captured_at"]}
if B.get("injury_pdf_b64"):
    stamp = B["injury_pdf_url"].rsplit("Injury-Report_", 1)[1].removesuffix(".pdf")
    rts = datetime.strptime(stamp, "%Y-%m-%d_%I_%M%p").replace(tzinfo=ET)
    # name directory: every player seen in 2024-27 (full names from 2026-27 boxes; initials from PBP)
    from nba_platform.core.names import name_key, _clean
    full = {}
    with gzip.open(f"{D}/s2627/nba_2026_27_pbp_box.jsonl.gz", "rt") as fh:
        for line in fh:
            bx = json.loads(line)["box"]
            for side in ("homeTeam", "awayTeam"):
                for p in bx[side]["players"]:
                    full[f"P{p['personId']}"] = (p.get("firstName", ""), p.get("familyName", ""))
    lastteam = pd.read_pickle(f"{D}/real_logs_v3.pkl").query("played == 1").sort_values("game_date_et").groupby("player_id")["team_id"].last().to_dict()
    # official team rosters (stats.nba.com commonteamroster, captured with the bundle): identifies players who
    # have not appeared in a box yet (rookies, two-way) and gives the CURRENT team (a roster beats an old box)
    for rr in (B.get("rosters") or []):
        pid_r = f"P{rr['PLAYER_ID']}"
        parts = str(rr.get("PLAYER", "")).split(" ", 1)
        if len(parts) == 2:
            full.setdefault(pid_r, (parts[0], parts[1]))
        lastteam[pid_r] = int(rr["TeamID"])
    ini = {}
    for x in ("2024", "2025"):
        if not os.path.exists(f"{D}/raw/cdnnba_{x}.csv"):
            continue
        n = pd.read_csv(f"{D}/raw/cdnnba_{x}.csv", usecols=["personId", "playerName", "playerNameI"], low_memory=False).dropna()
        for pid_, ln, ni in n.drop_duplicates("personId").itertuples(index=False):
            if pid_ > 0:
                ini[f"P{int(pid_)}"] = (str(ni).split(". ")[0][:1], str(ln))

    def directory(key, team_id):
        last, first = key.split("|")
        c = [p for p, (f_, l_) in full.items() if name_key(f_, l_) == key]
        if not c:
            c = [p for p, (i_, l_) in ini.items() if _clean(l_) == last and i_.lower() == first[:1]]
        c = [p for p in c if lastteam.get(p, team_id) == team_id] or (c if len(c) == 1 else [])
        if len(c) != 1:
            return None
        pid = c[0]
        f_, l_ = full.get(pid, (ini.get(pid, ("", ""))[0], ini.get(pid, ("", ""))[1]))
        s.x("""INSERT OR IGNORE INTO players(player_id, nba_person_id, full_name, first_name, last_name, name_key, created_at)
               VALUES (?,?,?,?,?,?,datetime('now'))""", (pid, int(pid[1:]), f"{f_} {l_}", f_, l_, key))
        return pid
    res = InjuryMonitor(s, directory).ingest_official(parse_pdf(base64.b64decode(B["injury_pdf_b64"])), rts, B["captured_at"])
    inj.update({"stored": res.stored, "unresolved": [r.player_raw for r in res.unresolved],
                "unmatched_games": sorted(set(res.unmatched_games)), "not_submitted": sorted(res.teams_not_submitted)})
# 5) pregame boxscores: CONFIRMED starters + official inactives (only if the league has posted them)
mon = InjuryMonitor(s)
for gid, box in (B.get("pregame_box") or {}).items():
    if not isinstance(box, dict):
        continue
    for side in ("homeTeam", "awayTeam"):
        t = box[side]; ps = t.get("players") or []
        st = [f"P{p['personId']}" for p in ps if str(p.get("starter")) == "1"]
        if len(st) == 5:
            s.insert_lineup(gid, int(t["teamId"]), "CONFIRMED", st, "nba_cdn_pregame_box", B["captured_at"])
        ina = [f"P{p['personId']}" for p in ps if p.get("status") == "INACTIVE"]
        if ina:
            mon.record_inactives(gid, int(t["teamId"]), ina, "nba_cdn_pregame_box", B["captured_at"])
s.commit()
# 6) champion anchor: NBA_TEAM_RATING_V2 blend
seg = pd.concat([pd.read_pickle(f"{D}/cdnnba_{x}_segments.pkl") for x in ("2024", "2025", "2026")])
cache = {}


def anchor(g, hi, ai):
    d = g["game_date_et"]
    if d not in cache:
        logs, games = load_frames(s, d)
        allg = pd.DataFrame([dict(r) for r in s.q("SELECT * FROM games")])
        av = logs[~logs["not_playing_reason"].fillna("").astype(str).str.startswith("INACTIVE")]
        mp = av.sort_values("game_date_et").groupby("player_id").tail(10).groupby("player_id")["minutes"].mean()
        cache[d] = (LineupRating(seg, games, d, RatingConfig(ridge=1500.0)), PointDiffRating(games, d),
                    rest_days(allg).set_index(["game_id", "team_id"])["rest"], mp)
    lr, pdr, rd, mp = cache[d]
    r = pilot.rates_for(d)
    pr = lr.predict({p: float(mp.get(p, 5.0)) for p in hi.available}, {p: float(mp.get(p, 5.0)) for p in ai.available},
                    r.expected_pace(hi.team_id, ai.team_id), rd.get((g["game_id"], hi.team_id), 2), rd.get((g["game_id"], ai.team_id), 2))
    return 0.295 * pdr.margin(hi.team_id, ai.team_id) + 0.845 * pr["margin"]


from nba_platform.injuries.play_rates import fit_status_play_rates
_pr_cache = {}


HIST = os.path.join(os.environ.get("NBA_CODE_DIR", "/home/claude/nba-platform"), "models/play_rates_hist.json")


def play_rates(d):
    """P(plays | official status) for weighting QUESTIONABLE/DOUBTFUL scenarios.
    1) our own archive of captured reports (games strictly before d) when n >= 40 per status;
    2) otherwise the historical NBA injury-report rates (models/play_rates_hist.json), fitted only on
       games before its 'fitted_through' date (must be <= d) and validated out of sample.
    Rates only apply to snapshots within `max_minutes_before_tip` (they were measured at that horizon);
    earlier windows stay CONDITIONAL. Missing status -> None -> CONDITIONAL."""
    if d not in _pr_cache:
        own = fit_status_play_rates(s, minutes_before_tip=90, min_n=40, before_date=d)
        out = {k: v for k, v in own.items() if v.get("p_play") is not None}
        meta = {"max_minutes_before_tip": 240, "source": "own_archive"}
        if os.path.exists(HIST):
            h = json.load(open(HIST))
            if h["fitted_through"] <= d:
                for k, v in h["rates"].items():
                    if k not in out and v.get("p_play") is not None:
                        out[k] = v
                meta = {**h.get("_meta", {}), "source": f"own_archive+PLAY_RATES_HIST@{h['fitted_through']}"}
        out["_meta"] = meta
        _pr_cache[d] = out
    return _pr_cache[d]


play_rates.source = "own_archive|PLAY_RATES_HIST"
pilot = Autopilot(s, cfg=MinutesConfig(**json.loads(a.minutes_cfg)), n_sims=a.sims, with_sim=True, sim_n=a.sims, anchor_margin_fn=anchor,
                  play_rates_fn=play_rates,
                  context_extra_fn=lambda g: {"sources": inj, "mode": ("REPLAY" if a.replay else "PROSPECTIVE") + ("" if a.stack == "STACK_V2_ENV" else "_SHADOW"), "calibration": json.load(open(os.path.join(os.environ.get("NBA_CODE_DIR", "/home/claude/nba-platform"), "models/calibration_beta_mkt.json")))["version"]},
                  stack_name=a.stack,
                  rates_cfg=(RatesConfig(**json.loads(a.rates_cfg)) if json.loads(a.rates_cfg) else None),
                  game_features=(pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]]
                                 if json.loads(a.minutes_cfg).get("blowout_feature") == "garbage"
                                 or json.loads(a.minutes_cfg).get("dnp_garbage") else None))
# Only the next `horizon-days` ET dates: forecasts weeks ahead have stale rosters/injuries and add noise to the archive.
dates = a.dates or sorted({r["game_date_et"] for r in s.q("SELECT game_date_et FROM games WHERE status=1 AND tip_utc > ?", (now.isoformat(),))})[:a.horizon_days]
out = []
for d in dates:
    for r in pilot.tick(now, d):
        out.append(r.__dict__)
print(json.dumps({"now": now.isoformat(), "injury": inj, "results": out, "seconds": round(time.time() - t0)}, default=str, indent=1))
