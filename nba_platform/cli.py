"""Command line entry points (what the cloud scheduler runs).

  python -m nba_platform.cli init      --db nba.db
  python -m nba_platform.cli daily     --db nba.db             # schedule + yesterday's finals
  python -m nba_platform.cli tick      --db nba.db             # every 5 min on game days
  python -m nba_platform.cli backtest  --db nba.db --from 2026-11-01 --to 2026-11-12
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from .core.timeutil import ET, game_date_et, parse_utc, utcnow
from .db.store import Store
from .injuries.play_rates import fit_status_play_rates
from .minutes.validate import backtest_baseline, backtest_engine, compare
from .pipeline import ingest
from .pipeline.autopilot import Autopilot, load_frames


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("nba_platform")
    ap.add_argument("cmd", choices=["init", "daily", "tick", "backtest"])
    ap.add_argument("--db", default="nba.db")
    ap.add_argument("--from", dest="d0")
    ap.add_argument("--to", dest="d1")
    ap.add_argument("--sims", type=int, default=10_000)
    a = ap.parse_args(argv)
    s = Store(a.db)
    if a.cmd == "init":
        ingest.ensure_teams(s)
        print(json.dumps({"ok": True}))
        return
    f = ingest.make_fetcher(s)
    now = utcnow()
    today = game_date_et(now)
    if a.cmd == "daily":
        ingest.ensure_teams(s)
        n = ingest.sync_schedule(s, f)
        fin = ingest.ingest_final_boxes(s, f, (today - timedelta(days=1)).isoformat())
        print(json.dumps({"schedule_games": n, "finals": fin}, default=str))
    elif a.cmd == "tick":
        rep = ingest.ingest_injury_report(s, f, now.astimezone(ET))
        pre = {}
        for g in s.games_on(today.isoformat()):
            mins_to_tip = (parse_utc(g["tip_utc"]) - now).total_seconds() / 60
            if g["status"] == 1 and 0 < mins_to_tip <= 90:     # starters are only posted close to tip
                pre[g["game_id"]] = ingest.ingest_pregame_box(s, f, g["game_id"])
        pilot = Autopilot(s, n_sims=a.sims,
                          play_rates_fn=lambda d: fit_status_play_rates(s, before_date=d))
        res = pilot.tick(now)
        print(json.dumps({"injury_report": rep, "pregame": pre, "refresh": [r.__dict__ for r in res]}, default=str))
    elif a.cmd == "backtest":
        d0, d1 = date.fromisoformat(a.d0), date.fromisoformat(a.d1)
        logs, games = load_frames(s, (d1 + timedelta(days=1)).isoformat())
        dates = sorted(x for x in logs["game_date_et"].unique() if d0.isoformat() <= x <= d1.isoformat())
        champ = backtest_engine(logs, games, dates, n_sims=2000)
        base = backtest_baseline(logs, dates)
        print(json.dumps({"engine": champ.metrics, "baseline": base.metrics, "compare": compare(base, champ)},
                         indent=1))


if __name__ == "__main__":
    main()
