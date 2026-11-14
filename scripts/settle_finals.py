"""V4: settle exactly the PUBLISHED pregame distribution of frozen snapshots against official final boxscores (JSONL archive).
  python scripts/settle_finals.py ARCHIVE.jsonl.gz DB.sqlite"""
import gzip, json, os, sys
sys.path.insert(0, os.environ.get("NBA_CODE_DIR", "/home/claude/nba-platform"))
from nba_platform.db.store import Store
from nba_platform.pipeline.settle import settle_published
from nba_platform.core.timeutil import iso_duration_minutes
arch, db = sys.argv[1], sys.argv[2]
s = Store(db)
out = []
with gzip.open(arch, "rt") as f:
    for line in f:
        g = json.loads(line); b = g["box"]
        if int(b.get("gameStatus", 0)) != 3 or not s.q("SELECT 1 FROM forecast_snapshots WHERE game_id=? LIMIT 1", (b["gameId"],)):
            continue
        rows = []
        for side in ("homeTeam", "awayTeam"):
            for p in b[side]["players"]:
                st = p.get("statistics") or {}
                played = int(str(p.get("played", "0")) == "1")
                rows.append({"player_id": f"P{p['personId']}", "team_id": b[side]["teamId"], "played": played,
                             "available": int(played or p.get("status") == "ACTIVE"),
                             "minutes": iso_duration_minutes(st.get("minutes")), "pts": st.get("points", 0) or 0,
                             "reb": st.get("reboundsTotal", 0) or 0, "ast": st.get("assists", 0) or 0,
                             "fg3m": st.get("threePointersMade", 0) or 0})
        s.upsert_game({"game_id": b["gameId"], **{k: v for k, v in dict(s.q("SELECT * FROM games WHERE game_id=?", (b["gameId"],))[0]).items() if k != "game_id"},
                       "status": 3, "home_score": int(b["homeTeam"]["score"]), "away_score": int(b["awayTeam"]["score"])})
        out.append({b["gameId"]: settle_published(s, b["gameId"], rows, int(b["homeTeam"]["score"]), int(b["awayTeam"]["score"]))})
print(json.dumps(out, indent=1))
