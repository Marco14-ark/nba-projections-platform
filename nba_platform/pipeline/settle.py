"""Prospective settlement: compare FROZEN pregame snapshots against official results.

Rules:
  * Only the latest snapshot created strictly before tip-off is settled (no post-tip forecasts).
  * Nothing is recomputed: predictions are read from the immutable snapshot tables.
  * Results rows are immutable (trigger) and keyed by snapshot, so re-running is idempotent.
  * Player props are settled for players who actually played (books void DNPs); minutes are
    settled for every player in the snapshot (DNP = 0 minutes).
"""
from __future__ import annotations

import json

from ..core.timeutil import utcnow
from ..db.store import Store


def pregame_snapshot(store: Store, game_id: str, tip_utc: str):
    r = store.q("""SELECT * FROM forecast_snapshots WHERE game_id=? AND created_at < ?
                   ORDER BY created_at DESC LIMIT 1""", (game_id, tip_utc))
    return r[0] if r else None


def settle_game(store: Store, game_id: str, official: list[dict], home_score: int, away_score: int) -> dict:
    """official: rows with player_id, played (0/1), minutes, pts, reb, ast, fg3m."""
    g = store.q("SELECT * FROM games WHERE game_id=?", (game_id,))
    if not g:
        return {"status": "UNKNOWN_GAME"}
    snap = pregame_snapshot(store, game_id, g[0]["tip_utc"])
    if snap is None:
        return {"status": "NO_PREGAME_SNAPSHOT"}
    sid = snap["snapshot_id"]
    if store.q("SELECT 1 FROM forecast_results WHERE snapshot_id=? LIMIT 1", (sid,)):
        return {"status": "ALREADY_SETTLED", "snapshot_id": sid}
    now = utcnow().isoformat()
    act = {r["player_id"]: r for r in official}
    n = 0
    # minutes (BASE / first scenario per team)
    for m in store.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND scenario IN ('BASE','S0')", (sid,)):
        a = act.get(m["player_id"])
        y = float(a["minutes"]) if a and a["played"] else 0.0
        store.x("INSERT INTO forecast_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, game_id, m["player_id"], "MIN", m["scenario"], m["mean"], m["median"], m["p10"], m["p90"],
                 None, None, y, abs(m["mean"] - y), int(m["p10"] <= y <= m["p90"]), now))
        n += 1
    stat = {"PTS": ("pts",), "REB": ("reb",), "AST": ("ast",), "3PM": ("fg3m",), "PRA": ("pts", "reb", "ast")}
    for p in store.q("""SELECT * FROM prop_forecasts WHERE snapshot_id=? AND scenario IN ('BASE','H0A0')
                        AND market IN ('PTS','REB','AST','3PM','PRA')""", (sid,)):
        a = act.get(p["player_id"])
        if not a or not a["played"]:
            continue
        y = float(sum(a[c] for c in stat[p["market"]]))
        store.x("INSERT INTO forecast_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, game_id, p["player_id"], p["market"], p["scenario"], p["mean"], p["median"], p["p10"], p["p90"],
                 p["p_over_fair"], p["model_fair_line"], y, abs(p["mean"] - y), int(p["p10"] <= y <= p["p90"]), now))
        n += 1
    gm = store.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,))
    if gm:
        payload = json.loads(gm[0][0])
        key = "BASE" if "BASE" in payload else sorted(payload)[0]
        mk = payload[key]["markets"]
        margin, total = home_score - away_score, home_score + away_score
        for name, pred, y, p10, p90 in (
                ("MARGIN", mk["margin"]["mean"], margin, mk["margin"]["p10"], mk["margin"]["p90"]),
                ("TOTAL", mk["total"]["mean"], total, mk["total"]["p10"], mk["total"]["p90"]),
                ("HOME_WIN", mk["p_home_win"], float(margin > 0), None, None)):
            store.x("INSERT INTO forecast_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, game_id, "", name, key, pred, None, p10, p90, None, None, y,
                     abs(pred - y), None if p10 is None else int(p10 <= y <= p90), now))
            n += 1
    store.commit()
    return {"status": "SETTLED", "snapshot_id": sid, "created_at": snap["created_at"], "rows": n}


# ================================================================================================
# V4 settlement: settle exactly the distribution that was published (see pipeline/publication.py)
# ================================================================================================
SETTLE_VERSION = "V4"
_STAT = {"PTS": ("pts",), "REB": ("reb",), "AST": ("ast",), "3PM": ("fg3m",), "PRA": ("pts", "reb", "ast")}


def category_of(context: dict) -> str:
    mode = str(context.get("mode", "PROSPECTIVE"))
    if mode.startswith("REPLAY"):
        return "REPLAY"
    if mode.endswith("_SHADOW") or context.get("stack") not in (None, "STACK_V2_ENV"):
        return "SHADOW"
    return "PROSPECTIVE"


def _publications(store: Store, sid: str) -> list[dict]:
    out = []
    for r in store.q("SELECT * FROM forecast_publications WHERE snapshot_id=?", (sid,)):
        d = dict(r)
        d["scenarios"] = json.loads(d["scenarios"])
        out.append(d)
    return out


def settle_published(store: Store, game_id: str, official: list[dict], home_score: int, away_score: int) -> dict:
    """official rows: player_id, team_id, available (0/1: dressed/active), played, minutes, pts, reb, ast, fg3m."""
    from .publication import backfill, matching_label

    g = store.q("SELECT * FROM games WHERE game_id=?", (game_id,))
    if not g:
        return {"status": "UNKNOWN_GAME"}
    snap = pregame_snapshot(store, game_id, g[0]["tip_utc"])
    if snap is None:
        return {"status": "NO_PREGAME_SNAPSHOT"}
    sid = snap["snapshot_id"]
    if snap["created_at"] >= g[0]["tip_utc"]:
        raise AssertionError("post-tip snapshot selected")            # defensive; pregame_snapshot filters
    if store.q("SELECT 1 FROM settlement_events WHERE snapshot_id=? AND settle_version=? LIMIT 1",
               (sid, SETTLE_VERSION)):
        return {"status": "ALREADY_SETTLED", "snapshot_id": sid}
    now = utcnow().isoformat()
    backfill(store, sid, now)                                         # no-op for V4 snapshots
    ctx = json.loads(snap["context"])
    cat = category_of(ctx)
    act = {r["player_id"]: r for r in official}
    avail = {p: bool(r.get("available", r["played"])) for p, r in act.items()}
    rows, events = 0, []

    def ins(comp, pid, market, pub, label, method, pred, y):
        nonlocal rows
        mean, med, p10, p90, pover, fair = pred
        store.x("INSERT INTO settlements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, game_id, cat, ctx.get("stack"), comp, pid, market, pub["publication"], label, method,
                 pub["weights_hash"], mean, med, p10, p90, pover, fair, y, abs(mean - y),
                 None if p10 is None else int(p10 <= y <= p90), SETTLE_VERSION, now))
        rows += 1

    for pub in _publications(store, sid):
        comp, team = pub["component"], pub["team_id"]
        if pub["publication"] == "CONDITIONAL":
            label = matching_label(pub, avail)
            if label is None:
                store.x("INSERT INTO settlement_events VALUES (?,?,?,?,?,?,?)",
                        (sid, comp, team, "NO_MATCHING_SCENARIO",
                         json.dumps({p: avail.get(p) for s in pub["scenarios"] for p in s["conditions"]}),
                         SETTLE_VERSION, now))
                events.append((comp, team, "NO_MATCHING_SCENARIO"))
                continue
            method = "CONDITIONAL_MATCHED"
        else:
            label = pub["published_label"]
            method = "PUBLISHED_" + pub["publication"]
        if comp == "MINUTES":
            for m in store.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND team_id=? AND scenario=?",
                             (sid, team, label)):
                a = act.get(m["player_id"])
                y = float(a["minutes"]) if a and a["played"] else 0.0
                ins(comp, m["player_id"], "MIN", pub, label, method,
                    (m["mean"], m["median"], m["p10"], m["p90"], None, None), y)
        elif comp == "PROPS":
            for p in store.q("""SELECT * FROM prop_forecasts WHERE snapshot_id=? AND scenario=?
                                AND market IN ('PTS','REB','AST','3PM','PRA')""", (sid, label)):
                a = act.get(p["player_id"])
                if not a or not a["played"]:
                    continue                                      # props on non-participants are void
                y = float(sum(a[c] for c in _STAT[p["market"]]))
                ins(comp, p["player_id"], p["market"], pub, label, method,
                    (p["mean"], p["median"], p["p10"], p["p90"], p["p_over_fair"], p["model_fair_line"]), y)
        elif comp == "GAME":
            payload = json.loads(store.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?",
                                         (sid,))[0][0])
            mk = payload[label]["markets"]
            margin, total = home_score - away_score, home_score + away_score
            for name, d, y in (("MARGIN", mk["margin"], margin), ("TOTAL", mk["total"], total)):
                ins(comp, "", name, pub, label, method, (d["mean"], d["median"], d["p10"], d["p90"], None, None), y)
            ins(comp, "", "HOME_WIN", pub, label, method, (mk["p_home_win"], None, None, None, None, None),
                float(margin > 0))
        store.x("INSERT INTO settlement_events VALUES (?,?,?,?,?,?,?)",
                (sid, comp, team, "SETTLED", json.dumps({"label": label, "method": method}), SETTLE_VERSION, now))
        events.append((comp, team, method, label))
    store.commit()
    return {"status": "SETTLED", "snapshot_id": sid, "category": cat, "rows": rows, "events": events}
