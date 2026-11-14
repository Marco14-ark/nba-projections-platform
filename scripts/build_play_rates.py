"""Historical P(plays | official status) from archived NBA injury reports (free, nba.com PDFs).
  python scripts/build_play_rates.py PART.json [PART.json ...] --fit-through 2026-10-01
Each part = {slot: {actual: 'YYYY-MM-DD_HH_MMAM', b64: pdf}} captured in the browser.
Outcome = did the player appear in that game (official boxscore). Rates are fitted on reports for games
strictly before --fit-through and validated out of sample on the later ones (2026-27)."""
import base64, glob, json, os, sys, re
sys.path.insert(0, "/home/claude/nba-platform")
import numpy as np, pandas as pd
from datetime import datetime
from nba_platform.core.timeutil import ET
from nba_platform.core.names import _clean
from nba_platform.sources.injury_report import parse_pdf
from nba_platform.sources.teams import BY_TRICODE
D = "/home/claude/data"
args = [a for i, a in enumerate(sys.argv[1:], 1) if a.endswith(".json") and sys.argv[i - 1] != "--fit-through"]
fit_through = sys.argv[sys.argv.index("--fit-through") + 1] if "--fit-through" in sys.argv else "2026-10-01"
L = pd.read_pickle(f"{D}/real_logs_v3.pkl"); G = pd.read_pickle(f"{D}/real_games_v3.pkl")
G["tip"] = pd.to_datetime(G.tip_utc, utc=True, format="ISO8601")
gidx = {(r.game_date_et, int(r.away_team_id), int(r.home_team_id)): (r.game_id, r.tip) for r in G.itertuples()}
# name directory: last name + first initial per (season, team) from play-by-play names
names = {}
for y in ("2024", "2025"):
    n = pd.read_csv(f"{D}/raw/cdnnba_{y}.csv", usecols=["personId", "playerName", "playerNameI"], low_memory=False).dropna()
    for pid, ln, ni in n.drop_duplicates("personId").itertuples(index=False):
        if pid > 0:
            names[f"P{int(pid)}"] = (str(ni).split(". ")[0][:1].lower(), _clean(str(ln)))
import gzip
for line in gzip.open(f"{D}/s2627/nba_2026_27_pbp_box.jsonl.gz", "rt"):
    bx = json.loads(line)["box"]
    for side in ("homeTeam", "awayTeam"):
        for p in bx[side]["players"]:
            names[f"P{p['personId']}"] = ((p.get("firstName") or " ")[0].lower(), _clean(p.get("familyName", "")))
L = L.merge(G[["game_id", "season"]], on="game_id")
season_of = dict(zip(G.game_id, G.season))
# resolve against the team's whole season roster (NOT that game's box): a player missing from the box
# simply did not play -> outcome 0. Resolving only within the box would bias P(play) upwards.
team_players = L.groupby(["season", "team_id"]).player_id.apply(set).to_dict()
played = {(g, p): int(v) for g, p, v in L[["game_id", "player_id", "played"]].itertuples(index=False)}
rows, unresolved, unmatched = [], 0, 0
for path in args:
    for slot, v in json.load(open(path)).items():
        rts = datetime.strptime(v["actual"], "%Y-%m-%d_%I_%M%p").replace(tzinfo=ET)
        for r in parse_pdf(base64.b64decode(v["b64"])):
            if r.not_yet_submitted or not r.player_raw:
                continue
            m, d_, y = r.game_date.split("/")
            gdate = f"{y}-{m}-{d_}"
            try:
                a_, h_ = r.matchup.split("@")
                key = (gdate, BY_TRICODE[a_][0], BY_TRICODE[h_][0])
            except (ValueError, KeyError):
                unmatched += 1; continue
            if key not in gidx:
                unmatched += 1; continue
            gid, tip = gidx[key]
            mbt = (tip - pd.Timestamp(rts)).total_seconds() / 60
            if mbt <= 0:
                continue                                      # report after tip: not a pregame status
            last, first = r.name_key.split("|")
            cands = [p for p in team_players.get((season_of[gid], r.team_id), set()) if names.get(p, ("", ""))[1] == last
                     and names[p][0] == first[:1]]
            if len(cands) != 1:
                unresolved += 1; continue
            rows.append({"game_id": gid, "date": gdate, "player_id": cands[0], "status": r.status.value,
                         "official": r.official_status, "minutes_before_tip": mbt, "played": played.get((gid, cands[0]), 0)})
R = pd.DataFrame(rows).sort_values("minutes_before_tip").drop_duplicates(["game_id", "player_id"], keep="first")
R.to_pickle(f"{D}/injury_status_outcomes.pkl")
tr, te = R[R.date < fit_through], R[R.date >= fit_through]
rates = {}
for s_, g in tr.groupby("status"):
    rates[s_] = {"n": int(len(g)), "p_play": round(float(g.played.mean()), 4) if len(g) >= 40 else None}
val = {}
for s_, g in te.groupby("status"):
    p = rates.get(s_, {}).get("p_play")
    if p is None:
        continue
    base = 0.5
    val[s_] = {"n": int(len(g)), "observed": round(float(g.played.mean()), 3), "predicted": p,
               "brier": round(float(((p - g.played) ** 2).mean()), 4), "brier_coinflip": round(float(((base - g.played) ** 2).mean()), 4)}
out = {"name": "PLAY_RATES_HIST", "fitted_through": fit_through, "n_reports_rows": int(len(R)), "unresolved_names": unresolved,
       "unmatched_games": unmatched, "minutes_before_tip_median": float(R.minutes_before_tip.median()),
       "minutes_before_tip_p90": float(R.minutes_before_tip.quantile(0.9)), "rates": rates, "validation_2026_27": val,
       "_meta": {"max_minutes_before_tip": 240}}
json.dump(out, open("/home/claude/nba-platform/models/play_rates_hist.json", "w"), indent=1)
print(json.dumps(out, indent=1))
