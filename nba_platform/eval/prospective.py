"""Prospective evaluation on REAL frozen snapshots (never backtests).

models():      per model DB, settled errors by component/market, and paired differences vs the
               Champion on exactly the same (game, player, market) keys, CI by bootstrap over games.
window_age():  error of the snapshots frozen at each refresh window (INITIAL/6H/90M/15M/30M) against
               the final box score, paired on the same keys. Uses each snapshot's PUBLISHED label
               (SINGLE or MIXTURE); CONDITIONAL publications are excluded (no single distribution)."""
from __future__ import annotations
import json
import sqlite3
from collections import defaultdict

import numpy as np

PROP_COLS = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "fg3m"}
WIN_ORDER = ["INITIAL", "6H", "90M", "30M", "15M"]


def _boot(diff_by_game: dict, n=2000, seed=1):
    g = list(diff_by_game)
    if len(g) < 2:
        return None
    S = np.array([sum(diff_by_game[k]) for k in g]); C = np.array([len(diff_by_game[k]) for k in g], float)
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    b = S[idx].sum(1) / C[idx].sum(1)
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def _settled(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = c.execute("""SELECT game_id, component, player_id, market, abs_error, in_p10_p90, pred_mean, actual
                        FROM settlements WHERE category != 'BACKTEST'""").fetchall()
    c.close()
    return {(g, comp, p, m): (e, cov, pm, a) for g, comp, p, m, e, cov, pm, a in rows if e is not None}


def models(dbs: dict[str, str], champion: str) -> dict:
    S = {name: _settled(p) for name, p in dbs.items()}
    base = S[champion]
    out = {}
    for name, s in S.items():
        by = defaultdict(list)
        for (g, comp, p, m), v in s.items():
            by[m if comp != "GAME" else f"GAME_{m}"].append((g, v))
        mk = {}
        for m, vals in sorted(by.items()):
            e = [v[0] for _, v in vals]
            cov = [v[1] for _, v in vals if v[1] is not None]
            row = {"n": len(e), "games": len({g for g, _ in vals}), "mae": round(float(np.mean(e)), 3),
                   "cov80": round(float(np.mean(cov)), 3) if cov else None}
            if name != champion:
                d = defaultdict(list)
                for (g, comp, p, mm), v in s.items():
                    if (mm if comp != "GAME" else f"GAME_{mm}") == m and (g, comp, p, mm) in base:
                        d[g].append(v[0] - base[(g, comp, p, mm)][0])
                if d:
                    row["paired_n"] = sum(len(x) for x in d.values())
                    row["d_mae_vs_champion"] = round(float(np.mean([x for v in d.values() for x in v])), 3)
                    row["d_mae_ci"] = _boot(d)
            mk[m] = row
        ng = len({k[0] for k in s})
        out[name] = {"games": ng, "markets": mk}
        if name != champion:
            out[name]["gate"] = promotion_gate(mk, ng)
    return out


KEY_MARKETS = ("MIN", "PTS", "REB", "AST", "PRA", "GAME_TOTAL", "GAME_MARGIN")
MIN_GAMES = 40


def promotion_gate(markets: dict, games: int) -> dict:
    """Never promotes anything by itself. 40 settled games is only the minimum to START evaluating;
    a shadow is a CANDIDATE_FOR_MANUAL_REVIEW only if no key market deteriorates significantly, at
    least two key markets improve significantly (paired CI entirely below 0), and the point estimate
    is not worse on PTS and PRA."""
    sig_better = [m for m in KEY_MARKETS if (markets.get(m) or {}).get("d_mae_ci") and markets[m]["d_mae_ci"][1] < 0]
    sig_worse = [m for m in KEY_MARKETS if (markets.get(m) or {}).get("d_mae_ci") and markets[m]["d_mae_ci"][0] > 0]
    core_ok = all((markets.get(m) or {}).get("d_mae_vs_champion", 1) <= 0 for m in ("PTS", "PRA"))
    if games < MIN_GAMES:
        st = "INSUFFICIENT_SAMPLE"
    elif sig_worse:
        st = "DETERIORATION"
    elif len(sig_better) >= 2 and core_ok:
        st = "CANDIDATE_FOR_MANUAL_REVIEW"
    else:
        st = "NO_CONSISTENT_GAIN"
    return {"status": st, "games": games, "min_games": MIN_GAMES, "significantly_better": sig_better,
            "significantly_worse": sig_worse, "auto_promotion": False}


def window_age(path: str) -> dict:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    fin = {r["game_id"]: dict(r) for r in c.execute("SELECT * FROM games WHERE status=3")}
    snaps = [dict(r) for r in c.execute("SELECT snapshot_id, game_id, refresh_window, created_at FROM forecast_snapshots")]
    snaps = [s for s in snaps if s["game_id"] in fin and s["created_at"] < fin[s["game_id"]]["tip_utc"]]
    pub = {(r["snapshot_id"], r["component"]): r for r in c.execute(
        "SELECT snapshot_id, component, publication, published_label FROM forecast_publications WHERE team_id=0")}
    logs = {(r["game_id"], r["player_id"]): dict(r) for r in c.execute(
        "SELECT * FROM player_game_logs WHERE game_id IN (%s)" % ",".join("?" * len(fin)), list(fin))} if fin else {}
    # latest snapshot per (game, window)
    last = {}
    for s in sorted(snaps, key=lambda x: x["created_at"]):
        last[(s["game_id"], s["refresh_window"])] = s
    err = defaultdict(dict)       # window -> {(game, player, market): abs error}
    excluded = defaultdict(int)
    for (gid, w), s in last.items():
        pp = pub.get((s["snapshot_id"], "PROPS"))
        if not pp or not pp["published_label"]:
            excluded[w] += 1
            continue
        for r in c.execute("SELECT player_id, market, mean, minutes_mean, p10, p90 FROM prop_forecasts WHERE snapshot_id=? AND scenario=?",
                           (s["snapshot_id"], pp["published_label"])):
            lg = logs.get((gid, r["player_id"]))
            if not lg or lg["played"] is None:
                continue
            if r["market"] in PROP_COLS:
                a = (lg[PROP_COLS[r["market"]]] or 0) if lg["played"] else 0
                err[w][(gid, r["player_id"], r["market"])] = abs(r["mean"] - a)
            if r["market"] == "PTS" and r["minutes_mean"] is not None:
                err[w][(gid, r["player_id"], "MIN")] = abs(r["minutes_mean"] - (lg["minutes"] if lg["played"] else 0))
        gp = pub.get((s["snapshot_id"], "GAME"))
        row = c.execute("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (s["snapshot_id"],)).fetchone()
        if gp and gp["published_label"] and row:
            P = json.loads(row["payload"]).get(gp["published_label"])
            g = fin[gid]
            if P and g["home_score"] is not None:
                m = P["markets"]
                err[w][(gid, "GAME", "MARGIN")] = abs(m["margin"]["mean"] - (g["home_score"] - g["away_score"]))
                err[w][(gid, "GAME", "TOTAL")] = abs(m["total"]["mean"] - (g["home_score"] + g["away_score"]))
    c.close()
    late = {}
    for k in set().union(*[set(v) for v in err.values()]) if err else []:
        for w in ("15M", "30M"):
            if k in err.get(w, {}):
                late[k] = err[w][k]
    if late:
        err["30M/15M"] = late
    out = {"by_window": {}, "paired": {}, "excluded_conditional": dict(excluded)}
    for w, e in err.items():
        by = defaultdict(list)
        for (g, p, m), v in e.items():
            by[m].append(v)
        out["by_window"][w] = {"games": len({k[0] for k in e}),
                               "markets": {m: {"n": len(v), "mae": round(float(np.mean(v)), 3)} for m, v in by.items()}}
    for a, b in (("6H", "90M"), ("90M", "30M/15M"), ("6H", "30M/15M"), ("INITIAL", "6H")):
        if a not in err or b not in err:
            continue
        common = set(err[a]) & set(err[b])
        res = {}
        for m in ("MIN", "PTS", "REB", "AST", "3PM", "MARGIN", "TOTAL"):
            d = defaultdict(list)
            for k in common:
                if k[2] == m:
                    d[k[0]].append(err[b][k] - err[a][k])
            if d:
                res[m] = {"n": sum(len(x) for x in d.values()), "games": len(d),
                          "d_mae": round(float(np.mean([x for v in d.values() for x in v])), 3), "ci": _boot(d)}
        out["paired"][f"{b} vs {a}"] = res
    return out
