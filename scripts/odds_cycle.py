"""Odds capture step (runs inside the cloud cycle, AFTER projections; never blocks them).

Decides by itself whether each provider should be queried now (own frequency + free-plan budget),
fetches real odds, matches events/players to the platform's games, and appends them to the odds
history DB. Without a key (repository secret) it records NO_KEY status and fetches nothing.
  python scripts/odds_cycle.py            (data dir from NBA_DATA_DIR)
Writes: <data>/odds/odds.db (append-only) and <data>/odds/odds_status.json (public, no secrets)."""
import json, os, sqlite3, sys, traceback
from datetime import datetime, timedelta, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nba_platform.odds.budget import allowance, can_spend
from nba_platform.odds.match import match_event, match_player, player_key_from_provider, team_index
from nba_platform.odds.providers import PROVIDERS
from nba_platform.odds.store import OddsStore
from nba_platform.sources.teams import TEAMS

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
NOW = datetime.fromisoformat(os.environ["ODDS_NOW"]) if os.environ.get("ODDS_NOW") else datetime.now(timezone.utc)
HORIZON_H = 18           # only games tipping within the next 18 h are worth spending budget on
MIN_GAP_MIN = 45         # never query a provider more often than this
CLOSE_WINDOW_MIN = 75    # near-close capture (for CLV) allowed once inside this window


def platform():
    c = sqlite3.connect(f"file:{os.path.join(D, 'live', 'prospective.db')}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    games = [dict(g) for g in c.execute("SELECT * FROM games WHERE tip_utc >= ? AND tip_utc <= ?",
                                        ((NOW - timedelta(hours=8)).isoformat(), (NOW + timedelta(hours=40)).isoformat()))]
    pkeys = {r["player_id"]: r["name_key"] for r in c.execute("SELECT player_id, name_key FROM players")}
    rosters = {}
    for g in games:
        s = c.execute("SELECT snapshot_id FROM forecast_snapshots WHERE game_id=? ORDER BY created_at DESC LIMIT 1",
                      (g["game_id"],)).fetchone()
        if s:
            pids = {r[0] for r in c.execute("SELECT DISTINCT player_id FROM prop_forecasts WHERE snapshot_id=?", (s[0],))}
            rosters[g["game_id"]] = {pkeys[p]: p for p in pids if pkeys.get(p)}
    return games, rosters


def policy(prov, store, pending, now):
    """-> (should_fetch: bool, reason, est_cost, budget)"""
    month, day = now.strftime("%Y-%m"), now.strftime("%Y-%m-%d")
    b = allowance(prov.monthly_limit, store.usage(prov.name, month), store.usage(prov.name, month, day), now)
    if not pending:
        return False, "NO_PENDING_GAMES", 0, b
    est = len(pending) if prov.unit == "objects" else len(prov.GAME_MARKETS)
    if not can_spend(b, est):
        return False, "BUDGET", est, b
    last = store.last_fetch(prov.name)
    first_tip = min(datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00")) for g in pending)
    mins_to_tip = (first_tip - now).total_seconds() / 60
    since = None if not last else (now - datetime.fromisoformat(last["requested_at"])).total_seconds() / 60
    sweeps = max(1, int(b["remaining_today"] // est))
    gap = min(240.0, max(MIN_GAP_MIN, mins_to_tip / sweeps))
    if since is None or since >= gap:
        return True, f"SCHEDULED(gap={gap:.0f}m)", est, b
    if mins_to_tip <= CLOSE_WINDOW_MIN and since >= 30:
        return True, "NEAR_CLOSE", est, b
    return False, f"WAIT(gap={gap:.0f}m,since={since:.0f}m)", est, b


def main():
    os.makedirs(os.path.join(D, "odds"), exist_ok=True)
    status = {"generated_at": NOW.isoformat(timespec="seconds"), "providers": {}}
    try:
        games, rosters = platform()
    except Exception as e:
        status["error"] = f"PLATFORM_DB:{type(e).__name__}"
        json.dump(status, open(os.path.join(D, "odds", "odds_status.json"), "w"), indent=1)
        print(json.dumps(status))
        return
    tidx = team_index([{"team_id": t[0], "tricode": t[1], "city": t[2], "name": t[3]} for t in TEAMS])
    pending = [g for g in games if g["status"] == 1 and
               NOW + timedelta(minutes=10) <= datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00")) <= NOW + timedelta(hours=HORIZON_H)]
    store = OddsStore(os.path.join(D, "odds", "odds.db"))
    for name, cls in PROVIDERS.items():
        prov = cls()
        st = {"configured": prov.configured, "unit": prov.unit, "monthly_limit": prov.monthly_limit}
        if not prov.configured:
            st["state"] = "NO_KEY"
            status["providers"][name] = st
            continue
        try:
            go, why, est, b = policy(prov, store, pending, NOW)
            st.update({"decision": why, "budget": {k: round(v, 1) if isinstance(v, float) else v for k, v in b.items()}})
            if go:
                tips = [datetime.fromisoformat(g["tip_utc"].replace("Z", "+00:00")) for g in pending]
                res = prov.fetch((min(tips) - timedelta(minutes=5)).isoformat(), (max(tips) + timedelta(minutes=5)).isoformat())
                events, ev_game = [], {}
                for eid, h, a, start in res.events:
                    gid, ms = match_event(h, a, start, games, tidx)
                    ev_game[str(eid)] = (gid, ms)
                    events.append({"provider_event_id": str(eid), "home_name": h, "away_name": a, "commence_utc": start,
                                   "game_id": gid, "match_status": ms})
                rows = []
                for q in res.quotes:
                    gid, ms = ev_game.get(q.provider_event_id, (None, "UNMATCHED_GAME"))
                    d = {**q.dict(), "game_id": gid, "match_status": ms, "player_key": None, "player_id": None}
                    if q.kind == "PLAYER":
                        d["player_key"] = q.provider_player_id or player_key_from_provider(q.player_name, None)
                        pk = player_key_from_provider(q.player_name, q.provider_player_id)
                        pid, pms = match_player(pk, rosters.get(gid, {})) if gid else (None, "UNMATCHED_GAME")
                        d["player_id"] = pid
                        if ms == "MATCHED":
                            d["match_status"] = pms
                    rows.append(d)
                rec = store.record(res, rows, events)
                st.update({"state": "OK" if res.ok else "ERROR", "fetched": True, "http": res.http_status,
                           "error": res.error, "cost": res.cost_units, "events": res.n_events, "quotes": len(rows),
                           "changes": rec["n_changes"],
                           "unmatched_players": sum(1 for r in rows if r["kind"] == "PLAYER" and r["player_id"] is None)})
            else:
                st.update({"state": "SKIPPED", "fetched": False})
        except Exception as e:      # an odds failure never propagates
            st.update({"state": "ERROR", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            traceback.print_exc()
        last = store.last_fetch(name)
        st["last_fetch"] = {k: last[k] for k in ("requested_at", "ok", "http_status", "error", "n_events", "n_quotes")} if last else None
        status["providers"][name] = st
    status["pending_games"] = len(pending)
    json.dump(status, open(os.path.join(D, "odds", "odds_status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
