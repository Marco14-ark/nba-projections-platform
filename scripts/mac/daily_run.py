"""$0 daily run (Mac, launchd). Steps stop early — never invent data — if a source is unreachable.
  NBA_DATA_DIR=<project>/data python scripts/mac/daily_run.py
1 connectivity  2 capture new finals  3 rebuild 2026-27 frames (official availability)  4 settle finals
5 bundle (schedule + latest official injury PDF + pregame boxscores <=2h to tip)  6 forecasts (snapshots)"""
import base64, json, os, subprocess, sys
from datetime import timedelta
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("NBA_CODE_DIR", ROOT)
D = os.environ.setdefault("NBA_DATA_DIR", os.path.join(ROOT, "data"))
import pandas as pd
from nba_platform.core.timeutil import ET, parse_utc, utcnow
from nba_platform.pipeline.capture import capture_finals, connectivity
from nba_platform.sources import nba_cdn
from nba_platform.sources.cdn_jsonl import build_frames
from nba_platform.sources.http import Fetcher, FetchError
from nba_platform.sources.injury_report import candidate_urls

log = {"started": utcnow().isoformat()}
f = Fetcher(retries=2)
c = connectivity(f); log["connectivity"] = c
if any(v != 200 for k, v in c.items() if k != "injury_pdf"):
    log["status"] = "STOPPED_SOURCE_UNREACHABLE"; print(json.dumps(log)); sys.exit(2)
arch = os.path.join(D, "s2627", "nba_2026_27_pbp_box.jsonl.gz"); os.makedirs(os.path.dirname(arch), exist_ok=True)
log["capture"] = capture_finals(f, arch)
G3, L3, B3, S3 = build_frames(arch)
G2, L2 = pd.read_pickle(f"{D}/real_games_v2.pkl"), pd.read_pickle(f"{D}/real_logs_v2.pkl")
pd.concat([G2, G3], ignore_index=True).to_pickle(f"{D}/real_games_v3.pkl")
pd.concat([L2, L3], ignore_index=True).to_pickle(f"{D}/real_logs_v3.pkl")
S3.to_pickle(f"{D}/cdnnba_2026_segments.pkl")
db = os.path.join(D, "live", "prospective.db"); os.makedirs(os.path.dirname(db), exist_ok=True)
py = sys.executable
log["settle"] = subprocess.run([py, os.path.join(ROOT, "scripts/settle_finals.py"), arch, db], capture_output=True, text=True).stdout[-2000:]
sched = json.loads(f.get(nba_cdn.SCHEDULE_URL, "nba_cdn").body)
games = [{"gameId": g["gameId"], "t": g["gameDateTimeUTC"], "h": g["homeTeam"]["teamId"], "a": g["awayTeam"]["teamId"],
          "ht": g["homeTeam"].get("teamTricode"), "at": g["awayTeam"].get("teamTricode"), "st": g["gameStatus"]}
         for d in sched["leagueSchedule"]["gameDates"] for g in d["games"] if g["gameId"].startswith("002")]
now = utcnow()
pdf, url = None, None
for u in candidate_urls(now.astimezone(ET), lookback_minutes=240):
    try:
        r = f.get(u, "nba_injury_report", ok_404=True)
    except FetchError:
        r = None
    if r is not None:
        pdf, url = base64.b64encode(r.body).decode(), u; break
pre = {}
for g in games:
    if g["st"] == 1 and timedelta(0) < parse_utc(g["t"]) - now <= timedelta(hours=2):
        try:
            r = f.get(nba_cdn.BOXSCORE_URL.format(game_id=g["gameId"]), "nba_cdn", ok_404=True)
            pre[g["gameId"]] = json.loads(r.body)["game"] if r is not None else "NOT_POSTED"
        except FetchError as e:
            pre[g["gameId"]] = f"ERR {e}"[:80]
bundle = {"captured_at": now.isoformat(), "injury_pdf_url": url, "injury_pdf_b64": pdf, "schedule": games, "pregame_box": pre}
bp = os.path.join(D, "live", f"bundle_{now.strftime('%Y%m%dT%H%M')}.json"); json.dump(bundle, open(bp, "w"))
fc = subprocess.run([py, os.path.join(ROOT, "scripts/live_forecast.py"), bp, db], capture_output=True, text=True)
log["forecast"] = fc.stdout[-3000:] if fc.returncode == 0 else ("FAILED " + fc.stderr[-1500:])
log["status"] = "OK" if fc.returncode == 0 else "FORECAST_FAILED"
print(json.dumps(log, default=str))
