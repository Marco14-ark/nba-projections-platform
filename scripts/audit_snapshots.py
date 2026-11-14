"""Audit every stored snapshot: pre-tip, publication kind, scenario weights, coherence, settlement.
Backfills publication records for pre-V4 snapshots (reconstructed only from stored rows).
  python scripts/audit_snapshots.py DB.sqlite [--json OUT]"""
import json, os, sys
sys.path.insert(0, os.environ.get("NBA_CODE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nba_platform.db.store import Store
from nba_platform.core.timeutil import utcnow
from nba_platform.pipeline.publication import backfill
from nba_platform.pipeline.settle import category_of

s = Store(sys.argv[1])
now = utcnow().isoformat()
rows = []
for snap in s.q("SELECT * FROM forecast_snapshots ORDER BY game_id, created_at"):
    sid = snap["snapshot_id"]
    backfill(s, sid, now)
    g = s.q("SELECT * FROM games WHERE game_id=?", (snap["game_id"],))[0]
    ctx = json.loads(snap["context"])
    pubs = {(r["component"], r["team_id"]): dict(r) for r in s.q(
        "SELECT * FROM forecast_publications WHERE snapshot_id=?", (sid,))}
    issues = []
    if snap["created_at"] >= g["tip_utc"]:
        issues.append("CREATED_AT_OR_AFTER_TIP")
    # minutes coherence: each team's published/conditional scenarios must sum to ~240 (+OT share)
    for (comp, team), p in pubs.items():
        if comp != "MINUTES":
            continue
        labels = [p["published_label"]] if p["published_label"] else [x["label"] for x in json.loads(p["scenarios"])]
        for lb in labels:
            tot = s.q("SELECT SUM(mean) FROM minutes_forecasts WHERE snapshot_id=? AND team_id=? AND scenario=?",
                      (sid, team, lb))[0][0] or 0.0
            if not (238.0 <= tot <= 250.0):
                issues.append(f"MINUTES_SUM:{team}:{lb}:{tot:.1f}")
        # OUT players must have no minutes rows
        side = "home" if ctx["home"]["team_id"] == team else "away"
        for pid in ctx[side]["out"]:
            if s.q("SELECT 1 FROM minutes_forecasts WHERE snapshot_id=? AND player_id=? LIMIT 1", (sid, pid)):
                issues.append(f"OUT_PLAYER_HAS_MINUTES:{pid}")
    gm = s.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,))
    if gm:
        for lb, v in json.loads(gm[0][0]).items():
            if lb != "MIX" and v["scenario"].get("home_prob") is None and v["scenario"].get("away_prob") is None:
                continue
    settled = [dict(r) for r in s.q("SELECT component, team_id, status, detail FROM settlement_events WHERE snapshot_id=?", (sid,))]
    rows.append({"snapshot_id": sid, "game_id": snap["game_id"], "window": snap["refresh_window"],
                 "created_at": snap["created_at"][:16], "tip_utc": g["tip_utc"][:16],
                 "category": category_of(ctx), "flag": snap["flag"], "superseded": snap["superseded_by"] is not None,
                 "publications": {f"{c}:{t}": p["publication"] for (c, t), p in sorted(pubs.items())},
                 "n_conditional": sum(p["publication"] == "CONDITIONAL" for p in pubs.values()),
                 "n_mixture": sum(p["publication"] == "MIXTURE" for p in pubs.values()),
                 "backfilled": all(p["backfilled"] for p in pubs.values()),
                 "issues": issues, "settlement": settled})
s.commit()
summary = {"snapshots": len(rows), "with_issues": sum(bool(r["issues"]) for r in rows),
           "any_post_tip": sum("CREATED_AT_OR_AFTER_TIP" in r["issues"] for r in rows),
           "with_conditional": sum(r["n_conditional"] > 0 for r in rows),
           "with_mixture": sum(r["n_mixture"] > 0 for r in rows),
           "categories": sorted({r["category"] for r in rows})}
out = {"summary": summary, "rows": rows}
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    if r["issues"]:
        print("ISSUE", r["snapshot_id"], r["issues"])
