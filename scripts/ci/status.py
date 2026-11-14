"""Write the public system-status file shown on the page (no secrets, no paths)."""
import json, os, re, sys
from datetime import datetime, timezone
out, probe_f, cycle_log = sys.argv[1], sys.argv[2], sys.argv[3]
steps = json.loads(os.environ.get("STEPS_JSON", "{}"))
probe = json.load(open(probe_f)) if os.path.exists(probe_f) else {}
cyc, frozen, settled, errors = {}, 0, 0, []
if os.path.exists(cycle_log):
    txt = open(cycle_log).read()
    i = txt.find("{")
    try:
        cyc = json.loads(txt[i:]) if i >= 0 else {}
    except json.JSONDecodeError:
        errors.append("cycle output not parseable")
    for cat, v in (cyc.get("forecast") or {}).items():
        frozen += len(re.findall(r'"status": "COMPUTED"', v))
        if v.startswith("FAILED"):
            errors.append(f"forecast {cat} failed")
    for cat, v in (cyc.get("settle") or {}).items():
        settled += len(re.findall(r'"status": "SETTLED"', v))
        if v.startswith("FAILED"):
            errors.append(f"settlement {cat} failed")
OPTIONAL = ("team roster (stats.nba.com)",)   # enhancement only: names/current team also come from boxscores
bad_sources = [k for k, v in probe.items() if isinstance(v, dict) and v.get("http") != 200]
errors += [f"source unavailable: {k}" for k in bad_sources if k not in OPTIONAL]
warnings = [f"optional source unavailable: {k}" for k in bad_sources if k in OPTIONAL]
OPTIONAL_STEPS = ("odds", "market_gate")      # sportsbook capture never blocks projections
errors += [f"step failed: {k}" for k, v in steps.items() if v not in ("success", "skipped") and k not in OPTIONAL_STEPS]
warnings += [f"optional step failed: {k}" for k, v in steps.items() if v not in ("success", "skipped") and k in OPTIONAL_STEPS]
odds_f = os.path.join(os.environ.get("NBA_DATA_DIR", ""), "odds", "odds_status.json")
odds = None
if os.path.exists(odds_f):
    o = json.load(open(odds_f))
    odds = {"checked_at": o.get("generated_at"), "pending_games": o.get("pending_games"),
            "providers": {k: {x: v.get(x) for x in ("configured", "state", "decision", "http", "error", "events", "quotes",
                                                     "unmatched_players", "last_fetch")} for k, v in (o.get("providers") or {}).items()}}
status = {"run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "run_url": f"{os.environ.get('GITHUB_SERVER_URL','')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}",
          "runner": "GitHub Actions (ubuntu-latest)", "steps": steps,
          "sources": {k: {"ok": v.get("http") == 200, "http": v.get("http")} for k, v in probe.items() if isinstance(v, dict)},
          "archive": cyc.get("archive"), "snapshots_frozen_this_run": frozen, "games_settled_this_run": settled,
          "odds": odds, "ok": not errors, "errors": errors, "warnings": warnings}
json.dump(status, open(out, "w"), indent=1)
print(json.dumps(status, indent=1))
