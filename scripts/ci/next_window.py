"""Next refresh-window target for the precision timer (GitHub cron is best-effort and was observed to
drift 30-200 min; a workflow that sleeps on a runner is exact to the second and free on public repos).

Targets per upcoming game: tip-355m (6H), tip-87m (90M), tip-28m (30M), tip-13m (15M) — each a few
minutes after the window opens so the cycle (≈3 min) freezes inside the window. Nearby targets
(< 6 min apart) are merged.  Usage: next_window.py SLATE_JSON [NOW_ISO] -> JSON {target, wait_s, game_ids}"""
import json, sys
from datetime import datetime, timedelta, timezone

OFFSETS = (355, 87, 28, 13)
MAX_WAIT_S = 5 * 3600 + 40 * 60      # stay under the 6 h job limit
MIN_LEAD_S = 60


def targets(slate: dict, now: datetime):
    out = []
    for g in slate.get("games", []):
        if g.get("status") not in (None, "SCHEDULED", "PREGAME"):
            continue
        tip = datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00"))
        for off in OFFSETS:
            t = tip - timedelta(minutes=off)
            if t > now + timedelta(seconds=MIN_LEAD_S):
                out.append((t, g["game_id"]))
    out.sort()
    merged = []
    for t, gid in out:
        if merged and (t - merged[-1][0]).total_seconds() < 360:
            merged[-1][1].append(gid)
        else:
            merged.append((t, [gid]))
    return merged


def main():
    slate = json.load(open(sys.argv[1]))
    now = datetime.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(timezone.utc)
    m = targets(slate, now)
    if not m:
        print(json.dumps({"target": None, "wait_s": None, "game_ids": []}))
        return
    t, gids = m[0]
    wait = int((t - now).total_seconds())
    print(json.dumps({"target": t.isoformat(), "wait_s": wait if wait <= MAX_WAIT_S else None,
                      "game_ids": gids, "beyond_limit": wait > MAX_WAIT_S}))


if __name__ == "__main__":
    main()
