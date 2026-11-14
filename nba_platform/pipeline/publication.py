"""What exactly was published before tip-off.

Rules (V4):
  * One availability scenario  -> SINGLE, the stored 'BASE' rows are the published distribution.
  * Several scenarios WITH justified probabilities (empirical play rates fitted on data strictly
    before the game date) -> MIXTURE; the stored mixture rows are published and the weights are
    frozen here (hash) in the same transaction as the snapshot.
  * Several scenarios WITHOUT justified probabilities -> CONDITIONAL: each scenario is a separate
    conditional forecast ("if X plays / if X is out"). No mixture is published. After the game the
    only scenario that can be settled is the one whose conditions actually happened.
"""
from __future__ import annotations

import hashlib
import json

from ..db.store import Store

COMPONENTS = ("MINUTES", "PROPS", "GAME")


def parse_conditions(name: str) -> dict[str, str]:
    """'P1:PLAYS+P2:OUT' -> {'P1': 'PLAYS', 'P2': 'OUT'};  'BASE' -> {}."""
    if not name or name == "BASE":
        return {}
    out = {}
    for part in name.split("+"):
        pid, st = part.rsplit(":", 1)
        if st not in ("PLAYS", "OUT"):
            raise ValueError(f"bad scenario condition {part!r}")
        out[pid] = st
    return out


def weights_hash(scen: list[dict]) -> str | None:
    if any(s["weight"] is None for s in scen):
        return None
    blob = json.dumps([[s["label"], round(float(s["weight"]), 12)] for s in scen], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _kind(scen: list[dict], mixture_label: str | None) -> tuple[str, str | None]:
    if len(scen) == 1:
        return "SINGLE", scen[0]["label"]
    if mixture_label and all(s["weight"] is not None for s in scen):
        return "MIXTURE", mixture_label
    return "CONDITIONAL", None


def record(store: Store, snapshot_id: str, component: str, team_id: int, scen: list[dict],
           mixture_label: str | None, weights_source: str | None, created_at: str, backfilled: bool = False) -> dict:
    if component not in COMPONENTS:
        raise ValueError(component)
    total = sum(s["weight"] for s in scen) if all(s["weight"] is not None for s in scen) else None
    if total is not None and len(scen) > 1 and abs(total - 1.0) > 1e-6:
        raise ValueError(f"scenario weights must sum to 1 (got {total})")
    kind, label = _kind(scen, mixture_label)
    rec = {"snapshot_id": snapshot_id, "component": component, "team_id": int(team_id), "publication": kind,
           "published_label": label, "scenarios": scen,
           "weights_source": weights_source if kind == "MIXTURE" else None,
           "weights_hash": weights_hash(scen) if kind == "MIXTURE" else None,
           "created_at": created_at, "backfilled": int(backfilled)}
    store.x("INSERT INTO forecast_publications VALUES (?,?,?,?,?,?,?,?,?,?)",
            (snapshot_id, component, int(team_id), kind, label, json.dumps(scen), rec["weights_source"],
             rec["weights_hash"], created_at, int(backfilled)))
    return rec


def team_scenarios(ctx_scen: list[dict]) -> list[dict]:
    """Minutes component: labels follow autopilot ('BASE' if one scenario else 'S{i}')."""
    if len(ctx_scen) == 1:
        return [{"label": "BASE", "conditions": parse_conditions(ctx_scen[0]["name"]), "weight": 1.0}]
    return [{"label": f"S{i}", "conditions": parse_conditions(s["name"]), "weight": s["prob"]}
            for i, s in enumerate(ctx_scen)]


def pair_scenarios(meta: dict) -> list[dict]:
    """Props/game component from the stored game_market payload (label -> scenario names/probs)."""
    out = []
    for label, v in meta.items():
        if label == "MIX":
            continue
        sc = v["scenario"]
        cond = {**parse_conditions(sc["home"]), **parse_conditions(sc["away"])}
        w = None if sc.get("home_prob") is None or sc.get("away_prob") is None else sc["home_prob"] * sc["away_prob"]
        out.append({"label": label, "conditions": cond, "weight": w})
    if len(out) > 1 and all(s["weight"] is not None for s in out):
        tot = sum(s["weight"] for s in out)
        for s in out:
            s["weight"] = s["weight"] / tot
    return out


def backfill(store: Store, snapshot_id: str, created_at: str) -> list[dict]:
    """Reconstruct publication records for a pre-V4 snapshot purely from what was stored."""
    if store.q("SELECT 1 FROM forecast_publications WHERE snapshot_id=? LIMIT 1", (snapshot_id,)):
        return []
    snap = store.q("SELECT * FROM forecast_snapshots WHERE snapshot_id=?", (snapshot_id,))[0]
    ctx = json.loads(snap["context"])
    recs = []
    for side in ("home", "away"):
        scen = team_scenarios(ctx["scenarios"][side])
        has_mix = bool(store.q("""SELECT 1 FROM minutes_forecasts WHERE snapshot_id=? AND team_id=? AND
                                  scenario='MIXTURE' LIMIT 1""", (snapshot_id, ctx[side]["team_id"])))
        recs.append(record(store, snapshot_id, "MINUTES", ctx[side]["team_id"], scen,
                           "MIXTURE" if has_mix else None, "pre-V4 minutes mixture" if has_mix else None,
                           created_at, backfilled=True))
    gm = store.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (snapshot_id,))
    if gm:
        meta = json.loads(gm[0][0])
        scen = pair_scenarios(meta)
        mix = "MIX" if "MIX" in meta else None
        for comp in ("PROPS", "GAME"):
            recs.append(record(store, snapshot_id, comp, 0, scen, mix, None, created_at, backfilled=True))
    return recs


def matching_label(pub: dict, available: dict[str, bool]) -> str | None:
    """CONDITIONAL settlement: the single scenario whose conditions all happened, else None."""
    hits = []
    for s in pub["scenarios"]:
        ok = all(pid in available and available[pid] == (st == "PLAYS") for pid, st in s["conditions"].items())
        if ok:
            hits.append(s["label"])
    return hits[0] if len(hits) == 1 else None
