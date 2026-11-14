"""Probe every NBA source once from the CI runner and record the real HTTP status (no data kept)."""
import json, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
     "Origin": "https://www.nba.com", "Referer": "https://www.nba.com/", "Accept": "application/json, text/plain, */*"}


def probe(url):
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=25) as r:
            return {"http": r.status, "bytes": len(r.read()), "s": round(time.time() - t, 2)}
    except Exception as e:  # noqa: BLE001
        return {"http": getattr(e, "code", 0), "error": type(e).__name__ + ": " + str(e)[:120], "s": round(time.time() - t, 2)}


et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
inj = None
for k in range(12):
    t = et - timedelta(minutes=15 * k); t = t.replace(minute=(t.minute // 15) * 15)
    r = probe(f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{t:%Y-%m-%d}_{t:%I}_{t:%M}{t:%p}.pdf")
    if r["http"] == 200:
        inj = r; break
    inj = r
out = {"probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "schedule (cdn.nba.com)": probe("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"),
       "boxscore (cdn.nba.com)": probe("https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600040.json"),
       "playbyplay (cdn.nba.com)": probe("https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022600040.json"),
       "injury report PDF (ak-static.cms.nba.com)": inj,
       "team roster (stats.nba.com)": probe("https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=2026-27&TeamID=1610612759")}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
