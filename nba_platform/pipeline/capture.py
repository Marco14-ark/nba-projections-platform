"""Free daily capture of 2026-27 data from NBA.com feeds (run on a machine whose network can reach
them; VERIFIED 2026-11-13 from the user's Mac with nba.com Referer/Origin headers: CDN schedule,
boxscore, play-by-play, stats.nba.com and the official injury PDF all returned 200).

Appends finished games to a JSON-lines archive ({gameId, actions, box}) — the same format the
browser capture produced — so `sources.cdn_jsonl` + `pbp_reconstruct` work unchanged.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

from ..core.timeutil import ET, utcnow
from ..sources import nba_cdn
from ..sources.http import Fetcher, FetchError
from ..sources.injury_report import candidate_urls

CHECKS = {
    "cdn_schedule": nba_cdn.SCHEDULE_URL,
    "cdn_scoreboard": nba_cdn.SCOREBOARD_URL,
    "stats_nba": "https://stats.nba.com/stats/scoreboardv2?GameDate={d}&LeagueID=00&DayOffset=0",
}


def connectivity(f: Fetcher) -> dict:
    out = {}
    d = utcnow().astimezone(ET).strftime("%Y-%m-%d")
    for k, u in CHECKS.items():
        try:
            r = f.get(u.format(d=d), "connectivity", headers={"x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}
                      if "stats.nba" in u else None)
            out[k] = r.status
        except FetchError as e:
            out[k] = f"FAIL {e}"[:120]
    for u in candidate_urls(utcnow().astimezone(ET), lookback_minutes=120):
        try:
            if f.get(u, "connectivity", ok_404=True) is not None:
                out["injury_pdf"] = 200
                break
        except FetchError as e:
            out["injury_pdf"] = f"FAIL {e}"[:120]
            break
    else:
        out.setdefault("injury_pdf", "NO_REPORT_IN_WINDOW")
    return out


def capture_finals(f: Fetcher, archive: str | Path) -> dict:
    archive = Path(archive)
    have = set()
    needs_newline = False
    if archive.exists():
        with gzip.open(archive, "rt") as fh:
            for line in fh:
                if line.strip():
                    have.add(json.loads(line)["gameId"])
                needs_newline = not line.endswith("\n")
    sched = json.loads(f.get(nba_cdn.SCHEDULE_URL, "nba_cdn").body)
    finals = [g["gameId"] for d in sched["leagueSchedule"]["gameDates"] for g in d["games"]
              if g["gameId"].startswith("002") and g.get("gameStatus") == 3 and g["gameId"] not in have]
    added, failed = 0, []
    with gzip.open(archive, "at") as out:
        if needs_newline and finals:      # archives written without a trailing newline must not glue records
            out.write("\n")
        for gid in finals:
            try:
                pbp = json.loads(f.get(nba_cdn.PBP_URL.format(game_id=gid), "nba_cdn").body)
                box = json.loads(f.get(nba_cdn.BOXSCORE_URL.format(game_id=gid), "nba_cdn").body)
                out.write(json.dumps({"gameId": gid, "actions": pbp["game"]["actions"], "box": box["game"]}) + "\n")
                added += 1
            except (FetchError, KeyError) as e:
                failed.append((gid, str(e)[:80]))
    return {"already": len(have), "added": added, "failed": failed, "at": datetime.now(ET).isoformat()}
