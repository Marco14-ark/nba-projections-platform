"""Server-side equivalent of scripts/mac/capture_bundle.js (for a free CI runner such as GitHub Actions).
  python scripts/fetch_capture.py OUT.json [--archive ARCHIVE.jsonl.gz]
Same sources (public nba.com JSON + official injury-report PDF), same gentle pacing, same output format.
On any non-200 the item is recorded as missing; nothing is invented. NOT VERIFIED until a real CI run succeeds."""
import base64, gzip, json, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
     "Origin": "https://www.nba.com", "Referer": "https://www.nba.com/", "Accept": "application/json, text/plain, */*"}


def get(url, binary=False, pause=0.4, timeout=30):
    time.sleep(pause)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            body = r.read()
            return r.status, body if binary else json.loads(body)
    except Exception as e:                       # noqa: BLE001
        return getattr(e, "code", 0), None


out_path = sys.argv[1]
known = set()
if "--archive" in sys.argv:
    for line in gzip.open(sys.argv[sys.argv.index("--archive") + 1], "rt"):
        if line.strip():
            known.add(json.loads(line)["gameId"])
now = datetime.now(timezone.utc)
st, sched = get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json")
if st != 200:
    sys.exit(f"schedule HTTP {st}")
games = [{"gameId": g["gameId"], "t": g["gameDateTimeUTC"], "h": g["homeTeam"]["teamId"], "a": g["awayTeam"]["teamId"],
          "ht": g["homeTeam"]["teamTricode"], "at": g["awayTeam"]["teamTricode"], "st": g["gameStatus"]}
         for d in sched["leagueSchedule"]["gameDates"] for g in d["games"] if g["gameId"].startswith("002")]
pdf = url = None
et = now.astimezone(ZoneInfo("America/New_York"))
for k in range(20):
    t = et - timedelta(minutes=15 * k); t = t.replace(minute=(t.minute // 15) * 15)
    u = f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{t:%Y-%m-%d}_{t:%I}_{t:%M}{t:%p}.pdf"
    st, body = get(u, binary=True, pause=0.5)
    if st == 200:
        pdf, url = base64.b64encode(body).decode(), u
        break
pre, finals, missing, live = {}, [], [], {}
for g in games:
    dt = (datetime.fromisoformat(g["t"].replace("Z", "+00:00")) - now).total_seconds() / 3600
    if g["st"] == 1 and 0 < dt <= 2.5:
        st, b = get(f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{g['gameId']}.json")
        pre[g["gameId"]] = b["game"] if st == 200 else "NOT_POSTED"
    if g["st"] == 2:
        st, b = get(f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{g['gameId']}.json")
        if st == 200:
            b = b["game"]
            live[g["gameId"]] = {"status": b["gameStatusText"], "period": b["period"], "clock": b["gameClock"],
                                 "home": b["homeTeam"]["score"], "away": b["awayTeam"]["score"], "captured_at": now.isoformat()}
    if g["st"] == 3 and g["gameId"] not in known:
        sb, b = get(f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{g['gameId']}.json")
        sp, p = get(f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{g['gameId']}.json")
        if sb == 200 and sp == 200:
            finals.append({"gameId": g["gameId"], "actions": p["game"]["actions"], "box": b["game"]})
        else:
            missing.append(g["gameId"])
rosters, roster_status = [], None
for tid in sorted({x for g in games for x in (g["h"], g["a"]) if x}):
    st, j = get(f"https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=2026-27&TeamID={tid}", timeout=12)
    if roster_status is None:
        roster_status = st
        if st != 200:          # stats.nba.com unreachable from this runner: stop instead of 30 timeouts
            break
    if st == 200:
        rs = j["resultSets"][0]
        for row in rs["rowSet"]:
            o = dict(zip(rs["headers"], row))
            rosters.append({k: o[k] for k in ("TeamID", "PLAYER_ID", "PLAYER", "POSITION")})
out = {"captured_at": now.isoformat(), "rosters": rosters, "injury_pdf_url": url, "injury_pdf_b64": pdf, "schedule": games,
       "pregame_box": pre, "finals": finals, "missing_finals": missing, "live": live}
json.dump(out, open(out_path, "w"))
print(json.dumps({"games": len(games), "pdf": url, "pregame": len(pre), "finals": len(finals), "missing": missing,
                  "live": len(live), "rosters": len(rosters), "roster_http": roster_status}))
