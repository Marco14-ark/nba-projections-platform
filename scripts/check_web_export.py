"""Validate the published web data against the frozen snapshots (read-only).
  python scripts/check_web_export.py WEB_DATA_DIR"""
import json, os, re, sqlite3, sys
D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
W = sys.argv[1]
c = sqlite3.connect(f"file:{D}/live/prospective.db?mode=ro", uri=True); c.row_factory = sqlite3.Row
problems, checked = [], {"games": 0, "props": 0, "injuries": 0}
for f in os.listdir(os.path.join(W, "games")):
    g = json.load(open(os.path.join(W, "games", f)))
    fc = g["forecast"]
    snap = c.execute("SELECT * FROM forecast_snapshots WHERE snapshot_id=?", (fc["snapshot_id"],)).fetchone()
    if snap is None:
        problems.append(f"{f}: snapshot not in DB"); continue
    if snap["created_at"] >= g["tip_utc"]:
        problems.append(f"{f}: post-tip snapshot")
    payload = json.loads(c.execute("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (fc["snapshot_id"],)).fetchone()[0])
    for gm in fc["game_markets"]:
        if abs(payload[gm["label"]]["markets"]["p_home_win"] - gm["markets"]["p_home_win"]) > 1e-4:
            problems.append(f"{f}: p_home_win mismatch {gm['label']}")
    for p in g["props"]["players"]:
        for mk, v in p["markets"].items():
            row = c.execute("SELECT mean FROM prop_forecasts WHERE snapshot_id=? AND player_id=? AND market=? AND scenario=?",
                            (fc["snapshot_id"], p["player_id"], mk, p["scenario"])).fetchone()
            if row is None or abs(row[0] - v["mean"]) > 0.006:
                problems.append(f"{f}: prop mismatch {p['player_id']} {mk}")
            checked["props"] += 1
        # player must be in that team's available list in the snapshot context
        ctx = json.loads(snap["context"])
        side = "home" if p["home"] else "away"
        if p["player_id"] not in ctx["available"][side]:
            problems.append(f"{f}: {p['player_id']} not in snapshot roster")
    for side in ("home", "away"):
        for i in fc["injuries"][side]:
            checked["injuries"] += 1
            if not (i["source"] and i["report_ts"] and i["observed_at"]):
                problems.append(f"{f}: injury without source/timestamp {i['player_id']}")
    checked["games"] += 1
blob = "".join(open(os.path.join(r, x)).read() for r, _, fs in os.walk(W) for x in fs)
for pat in (r"/home/", r"/Users/", r"@gmail", r"\.db\b", r"token", r"password"):
    if re.search(pat, blob, re.I):
        problems.append(f"private-looking pattern in export: {pat}")
print(json.dumps({"checked": checked, "problems": problems[:20], "n_problems": len(problems)}, indent=1))
