"""Data-integrity checks used by the prospective cycle and the tests."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..core.timeutil import iso_duration_minutes


def archive_integrity(path: str | Path) -> dict:
    raw = gzip.open(path, "rt").read() if Path(path).exists() else ""
    lines = [l for l in raw.split("\n") if l.strip()]
    ids, bad = [], 0
    for l in lines:
        try:
            ids.append(json.loads(l)["gameId"])
        except (json.JSONDecodeError, KeyError):
            bad += 1
    return {"records": len(lines), "unique_games": len(set(ids)), "duplicates": len(ids) - len(set(ids)),
            "bad_json": bad, "missing_trailing_newline": bool(raw) and not raw.endswith("\n")}


def box_coherence(box: dict, tol_min: float = 1.0) -> list[str]:
    """Final boxscore: team minutes = 240 + 25*OT and player points sum = team score."""
    issues = []
    periods = int(box.get("period", 4) or 4)
    expect = 240.0 + 25.0 * max(0, periods - 4)
    for side in ("homeTeam", "awayTeam"):
        t = box[side]
        mins = sum(iso_duration_minutes((p.get("statistics") or {}).get("minutes")) for p in t["players"])
        pts = sum(int((p.get("statistics") or {}).get("points", 0) or 0) for p in t["players"])
        if abs(mins - expect) > tol_min:
            issues.append(f"{side}:MINUTES_SUM:{mins:.1f}!={expect:.0f}")
        if pts != int(t.get("score", -1)):
            issues.append(f"{side}:POINTS_SUM:{pts}!={t.get('score')}")
    return issues
