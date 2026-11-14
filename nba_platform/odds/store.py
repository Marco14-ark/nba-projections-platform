"""Append-only odds history (SQLite). Rows are never updated or deleted: a quote row is written when
it first appears or when its line/price/availability changes; a 'withdrawn' row (available=0, price NULL)
is written when a previously offered quote disappears from a successful fetch of the same event.
The state at any time t is the latest row per key with captured_at <= t (exact closing lines / CLV)."""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_fetches (
  fetch_id TEXT PRIMARY KEY, provider TEXT NOT NULL, endpoint TEXT, requested_at TEXT NOT NULL,
  http_status INTEGER, ok INTEGER NOT NULL, error TEXT, cost_units INTEGER NOT NULL DEFAULT 0,
  n_events INTEGER, n_quotes INTEGER, n_changes INTEGER, raw_sha256 TEXT, headers TEXT, synthetic INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS odds_events (
  fetch_id TEXT NOT NULL, provider TEXT NOT NULL, provider_event_id TEXT NOT NULL, home_name TEXT, away_name TEXT,
  commence_utc TEXT, game_id TEXT, match_status TEXT NOT NULL, PRIMARY KEY (fetch_id, provider_event_id));
CREATE TABLE IF NOT EXISTS odds_quotes (
  fetch_id TEXT NOT NULL, captured_at TEXT NOT NULL, provider TEXT NOT NULL, book TEXT NOT NULL,
  provider_event_id TEXT NOT NULL, game_id TEXT, kind TEXT NOT NULL, market TEXT NOT NULL, team_side TEXT,
  player_key TEXT, player_id TEXT, player_name TEXT, side TEXT NOT NULL, line REAL, price INTEGER,
  available INTEGER NOT NULL, book_updated_at TEXT, match_status TEXT NOT NULL, notes TEXT);
CREATE INDEX IF NOT EXISTS ix_oq_game ON odds_quotes(game_id, captured_at);
CREATE INDEX IF NOT EXISTS ix_oq_key ON odds_quotes(provider, book, provider_event_id, market, player_key, side);
CREATE TRIGGER IF NOT EXISTS trg_oq_upd BEFORE UPDATE ON odds_quotes BEGIN SELECT RAISE(ABORT,'odds_quotes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_oq_del BEFORE DELETE ON odds_quotes BEGIN SELECT RAISE(ABORT,'odds_quotes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_of_upd BEFORE UPDATE ON odds_fetches BEGIN SELECT RAISE(ABORT,'odds_fetches are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_of_del BEFORE DELETE ON odds_fetches BEGIN SELECT RAISE(ABORT,'odds_fetches are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_oe_upd BEFORE UPDATE ON odds_events BEGIN SELECT RAISE(ABORT,'odds_events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_oe_del BEFORE DELETE ON odds_events BEGIN SELECT RAISE(ABORT,'odds_events are append-only'); END;
"""

KEY = ("provider", "book", "provider_event_id", "kind", "market", "team_side", "player_key", "side")


class OddsStore:
    def __init__(self, path: str = ":memory:"):
        self.c = sqlite3.connect(path)
        self.c.row_factory = sqlite3.Row
        self.c.executescript(SCHEMA)

    def q(self, sql, args=()):
        return self.c.execute(sql, args).fetchall()

    def current(self, provider: str | None = None, game_id: str | None = None, at: str | None = None) -> list[dict]:
        """Latest row per quote key (optionally as of time `at`); withdrawn quotes are excluded."""
        w, a = ["1=1"], []
        if provider:
            w.append("provider=?"); a.append(provider)
        if game_id:
            w.append("game_id=?"); a.append(game_id)
        if at:
            w.append("captured_at<=?"); a.append(at)
        rows = self.q(f"""SELECT * FROM odds_quotes WHERE {' AND '.join(w)} ORDER BY captured_at, rowid""", a)
        last = {}
        for r in rows:
            last[tuple(r[k] for k in KEY)] = dict(r)
        return [r for r in last.values() if r["available"]]

    def record(self, res, matched: list[dict], events: list[dict], synthetic: bool = False) -> dict:
        """res: providers.FetchResult. matched: dict rows (quote fields + game_id, player_id, match_status).
        events: [{provider_event_id, home_name, away_name, commence_utc, game_id, match_status}]"""
        fid = f"{res.provider}:{res.requested_at}:{uuid.uuid4().hex[:8]}"
        n_changes = 0
        if res.ok:
            prev = {}
            for r in self.q("SELECT * FROM odds_quotes WHERE provider=? ORDER BY captured_at, rowid", (res.provider,)):
                prev[tuple(r[k] for k in KEY)] = dict(r)
            seen_events = {e["provider_event_id"] for e in events}
            now_keys = set()
            for m in matched:
                k = tuple(m.get(x) or ("" if x in ("team_side", "player_key") else None) for x in KEY)
                m["team_side"] = m.get("team_side") or ""
                m["player_key"] = m.get("player_key") or ""
                now_keys.add(k)
                p = prev.get(k)
                if p and p["available"] and p["line"] == m["line"] and p["price"] == m["price"] \
                        and bool(p["available"]) == bool(m.get("available", True)) and p["game_id"] == m.get("game_id"):
                    continue
                self._ins(fid, res.requested_at, m)
                n_changes += 1
            for k, p in prev.items():
                if p["available"] and p["provider_event_id"] in seen_events and k not in now_keys:
                    self._ins(fid, res.requested_at, {**p, "price": None, "available": 0, "notes": json.dumps(["WITHDRAWN"])})
                    n_changes += 1
            for e in events:
                self.c.execute("INSERT OR IGNORE INTO odds_events VALUES (?,?,?,?,?,?,?,?)",
                               (fid, res.provider, e["provider_event_id"], e.get("home_name"), e.get("away_name"),
                                e.get("commence_utc"), e.get("game_id"), e.get("match_status", "UNMATCHED")))
        self.c.execute("INSERT INTO odds_fetches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (fid, res.provider, res.endpoint, res.requested_at, res.http_status, int(res.ok), res.error,
                        res.cost_units, res.n_events, len(matched), n_changes, res.raw_sha256,
                        json.dumps(res.headers or {}), int(synthetic)))
        self.c.commit()
        return {"fetch_id": fid, "n_quotes": len(matched), "n_changes": n_changes}

    def _ins(self, fid, ts, m):
        notes = m.get("notes")
        if isinstance(notes, list):
            notes = json.dumps(notes)
        self.c.execute("""INSERT INTO odds_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (fid, ts, m["provider"], m["book"], m["provider_event_id"], m.get("game_id"), m["kind"],
                        m["market"], m.get("team_side") or "", m.get("player_key") or "", m.get("player_id"),
                        m.get("player_name"), m["side"], m.get("line"), m.get("price"),
                        int(bool(m.get("available", True))), m.get("book_updated_at"),
                        m.get("match_status", "UNMATCHED"), notes))

    def usage(self, provider: str, month: str, day: str | None = None) -> int:
        sql = "SELECT COALESCE(SUM(cost_units),0) FROM odds_fetches WHERE provider=? AND synthetic=0 AND substr(requested_at,1,7)=?"
        a = [provider, month]
        if day:
            sql += " AND substr(requested_at,1,10)=?"
            a.append(day)
        return int(self.c.execute(sql, a).fetchone()[0])

    def last_fetch(self, provider: str) -> dict | None:
        r = self.q("SELECT * FROM odds_fetches WHERE provider=? AND synthetic=0 ORDER BY requested_at DESC LIMIT 1", (provider,))
        return dict(r[0]) if r else None
