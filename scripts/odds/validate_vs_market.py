"""Market gate: may the value engine ever say VALUE for a market?

For every settled game with captured odds, take the CLOSING real two-way price (last quote captured
before tip) and the frozen pregame model distribution that was settled, and compare the model's
probability with the book's no-vig probability on the realised outcome (pushes excluded).
A market becomes VALIDATED only with >= MIN_N priced outcomes AND the model beating the no-vig
market on log loss with a bootstrap 95% interval entirely below zero. Otherwise NOT_VALIDATED.
Writes <data>/odds/market_gate.json (+ a public copy in the export)."""
import json, os, sqlite3, sys
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nba_platform.odds.markets import GAME_MARKETS, PLAYER_MARKETS
from nba_platform.odds.pricing import no_vig_pair
from nba_platform.odds.store import OddsStore
from nba_platform.odds.value import side_probs

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
MIN_N = 300


def main():
    out = {m: {"status": "NOT_VALIDATED", "n": 0} for m in PLAYER_MARKETS + GAME_MARKETS}
    odb = os.path.join(D, "odds", "odds.db")
    if os.path.exists(odb):
        st = OddsStore(odb)
        c = sqlite3.connect(f"file:{os.path.join(D, 'live', 'prospective.db')}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        obs = defaultdict(list)       # market -> [(p_model, p_novig, y)]
        for g in c.execute("SELECT * FROM games WHERE status=3 AND game_id IN (SELECT DISTINCT game_id FROM settlements WHERE category='PROSPECTIVE')"):
            qs = [q for q in st.current(game_id=g["game_id"], at=g["tip_utc"]) if q["match_status"] in ("MATCHED", "MATCHED_FUZZY")]
            if not qs:
                continue
            sets = {(s["component"], s["player_id"], s["market"]): dict(s) for s in
                    c.execute("SELECT * FROM settlements WHERE game_id=? AND category='PROSPECTIVE'", (g["game_id"],))}
            any_s = next(iter(sets.values()))
            sid, lab = any_s["snapshot_id"], any_s["settled_label"]
            gd = {r["market"]: (r["lo"], json.loads(r["counts"])) for r in
                  c.execute("SELECT * FROM game_distributions WHERE snapshot_id=? AND scenario=?", (sid, lab))}
            margin, total = g["home_score"] - g["away_score"], g["home_score"] + g["away_score"]
            pairs = defaultdict(dict)
            for q in qs:
                pairs[(q["book"], q["market"], q["team_side"], q["player_id"] or "", abs(q["line"] or 0))][q["side"]] = q
            for (book, mk, ts, pid, _), sd in pairs.items():
                a, b = ("over", "under") if mk not in ("ML", "SPREAD") else ("home", "away")
                if a not in sd or b not in sd:
                    continue
                qa, qb = sd[a], sd[b]
                line = qa["line"]
                if mk == "ML":
                    y = 1.0 if margin > 0 else 0.0
                elif mk == "SPREAD":
                    v = margin + line
                    y = None if v == 0 else float(v > 0)
                elif mk in ("TOTAL", "TEAM_TOTAL"):
                    act = total if mk == "TOTAL" else (g["home_score"] if ts == "home" else g["away_score"])
                    y = None if act == line else float(act > line)
                else:
                    s = sets.get(("PROPS", pid, mk))
                    if not s or s["actual"] is None:
                        continue
                    y = None if s["actual"] == line else float(s["actual"] > line)
                if y is None:
                    continue
                if mk in PLAYER_MARKETS:
                    dr = c.execute("SELECT lo, counts FROM prop_distributions WHERE snapshot_id=? AND player_id=? AND scenario=? AND market=?",
                                   (sid, pid, lab, mk)).fetchone()
                    if not dr:
                        continue
                    pr = side_probs(mk, "over", line, {mk: (dr[0], json.loads(dr[1]))})
                else:
                    if not gd:
                        continue
                    pr = side_probs(mk, f"{ts}:over" if mk == "TEAM_TOTAL" else a, line, gd)
                p = pr["win"] / max(1e-9, 1 - pr["push"])
                nv = no_vig_pair(qa["price"], qb["price"])[0]
                obs[mk].append((p, nv, y))
        for mk, v in obs.items():
            A = np.array(v)
            p, q, y = np.clip(A[:, 0], 1e-4, 1 - 1e-4), np.clip(A[:, 1], 1e-4, 1 - 1e-4), A[:, 2]
            lm = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            lq = -(y * np.log(q) + (1 - y) * np.log(1 - q))
            d = lm - lq
            bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(1000)] if len(d) > 1 else [0, 0]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            ok = len(d) >= MIN_N and ci[1] < 0
            out[mk] = {"status": "VALIDATED" if ok else "NOT_VALIDATED", "n": int(len(d)),
                       "logloss_model": float(lm.mean()), "logloss_market_novig": float(lq.mean()),
                       "diff_ci95": ci, "brier_model": float(np.mean((p - y) ** 2)), "brier_market": float(np.mean((q - y) ** 2))}
    gate = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "min_n": MIN_N,
            "rule": "model log loss < closing no-vig log loss (bootstrap 95% CI < 0) with n >= min_n", "markets": out}
    os.makedirs(os.path.join(D, "odds"), exist_ok=True)
    json.dump(out, open(os.path.join(D, "odds", "market_gate.json"), "w"), indent=1)
    json.dump(gate, open(os.path.join(D, "odds", "market_gate_report.json"), "w"), indent=1)
    print(json.dumps({k: (v["status"], v["n"]) for k, v in out.items()}))


if __name__ == "__main__":
    main()
