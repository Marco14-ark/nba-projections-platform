"""One prospective cycle from an already-captured browser bundle (no network needed here).
  python scripts/prospective_cycle.py CAPTURE.json [--no-forecast] [--no-shadow]
CAPTURE = {"captured_at", "injury_pdf_url", "injury_pdf_b64", "schedule", "pregame_box", "finals": [ {gameId, actions, box} ]}
Steps: 1 append new finals to the archive (newline-safe, de-duplicated)  2 rebuild 2026-27 frames, segments,
garbage time  3 settle PUBLISHED distributions (champion DB and shadow DB separately)  4 freeze new
snapshots (champion = PROSPECTIVE; challenger = SHADOW, separate DB). Categories are never mixed."""
import gzip, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
import pandas as pd
from nba_platform.sources.cdn_jsonl import build_frames
from nba_platform.minutes.garbage import game_garbage

cap_path = sys.argv[1]
C = json.load(open(cap_path))
log = {"capture": cap_path, "captured_at": C["captured_at"]}
arch = os.path.join(D, "s2627", "nba_2026_27_pbp_box.jsonl.gz")
# 1) archive append: rewrite whole file (atomic) so a missing trailing newline can never glue records
raw = gzip.open(arch, "rt").read() if os.path.exists(arch) else ""
recs = [json.loads(l) for l in raw.split("\n") if l.strip()]
have = {r["gameId"] for r in recs}
dups = len(recs) - len(have)
new = [g for g in C.get("finals", []) if int(g["box"].get("gameStatus", 0)) == 3 and g["gameId"] not in have]
if new:
    with gzip.open(arch + ".tmp", "wt") as f:
        for r in recs + new:
            f.write(json.dumps(r) + "\n")
    os.replace(arch + ".tmp", arch)
log["archive"] = {"before": len(recs), "added": len(new), "duplicates_in_archive": dups}
# live scores (display only, timestamped; not a model input)
json.dump({"captured_at": C["captured_at"], "games": C.get("live", {})},
          open(os.path.join(D, "live", "live_scores.json"), "w"))
# 2) frames
G3, L3, B3, S3 = build_frames(arch)
G2, L2 = pd.read_pickle(f"{D}/real_games_v2.pkl"), pd.read_pickle(f"{D}/real_logs_v2.pkl")
pd.concat([G2, G3], ignore_index=True).to_pickle(f"{D}/real_games_v3.pkl")
pd.concat([L2, L3], ignore_index=True).to_pickle(f"{D}/real_logs_v3.pkl")
S3.to_pickle(f"{D}/cdnnba_2026_segments.pkl")
gb = pd.concat([game_garbage(pd.read_pickle(f"{D}/cdnnba_{y}_segments.pkl")) for y in ("2024", "2025", "2026")])
gb.to_pickle(f"{D}/garbage_by_game.pkl")
log["frames"] = {"games_2026_27": int(len(G3)), "garbage_rows": int(len(gb))}
py = sys.executable
# 3) settlement (published distributions only), each category in its own DB
dbs = {"PROSPECTIVE": os.path.join(D, "live", "prospective.db")}
SHADOWS = {"STACK_V3_SIGMA": {"sigma_dof_correct": True, "sigma_dof_scope": "starters"},
           "STACK_V4_BLOWOUT": {"blowout_feature": "garbage", "dnp_garbage": True},
           "STACK_V5_PACE": {},
           "STACK_V6_MINROLE": {"start_mix": True},
           "STACK_V9_COHERENT": {},                            # V11 MIN_V3_COHERENT: rotation script and final score coherent
           "STACK_V8_TEAMVAR": {}}                             # V11 SHOT_V3_TEAMVAR: team-game opportunity shocks
SHADOW_RATES = {"STACK_V5_PACE": {"pace_norm": "team_mean"}}     # V9: pace without double-counted environment
SHADOW_JOINT = {"STACK_V9_COHERENT": {"script_coherent_sd": 8.0},
                "STACK_V8_TEAMVAR": {"team_use_sd": 0.0415, "team_tov_sd": 0.0612, "team_trip_sd": 0.0686}}
if "--no-shadow" not in sys.argv:
    for name in SHADOWS:
        dbs["SHADOW:" + name] = os.path.join(D, "live", f"shadow_{name}.db")
log["settle"] = {}
for cat, db in dbs.items():
    if not os.path.exists(db):
        continue
    r = subprocess.run([py, os.path.join(ROOT, "scripts/settle_finals.py"), arch, db], capture_output=True, text=True)
    log["settle"][cat] = r.stdout[-1500:] if r.returncode == 0 else "FAILED " + r.stderr[-800:]
# 4) freeze new snapshots
if "--no-forecast" not in sys.argv:
    bundle = {k: C.get(k) for k in ("captured_at", "injury_pdf_url", "injury_pdf_b64", "schedule", "pregame_box", "rosters")}
    bp = os.path.join(D, "live", "bundle_" + C["captured_at"][:16].replace(":", "").replace("-", "") + ".json")
    json.dump(bundle, open(bp, "w"))
    runs = {"PROSPECTIVE": [bp, dbs["PROSPECTIVE"]]}
    for cat, db in dbs.items():
        if cat.startswith("SHADOW:"):
            name = cat.split(":", 1)[1]
            runs[cat] = [bp, db, "--stack", name, "--minutes-cfg", json.dumps(SHADOWS[name]),
                         "--rates-cfg", json.dumps(SHADOW_RATES.get(name, {})),
                         "--joint-cfg", json.dumps(SHADOW_JOINT.get(name, {}))]
    log["forecast"] = {}
    for cat, args in runs.items():
        r = subprocess.run([py, os.path.join(ROOT, "scripts/live_forecast.py"), *args], capture_output=True, text=True)
        log["forecast"][cat] = r.stdout[-2500:] if r.returncode == 0 else "FAILED " + r.stderr[-1500:]
print(json.dumps(log, indent=1, default=str))
