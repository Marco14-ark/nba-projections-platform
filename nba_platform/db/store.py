"""Repository layer. SQLite implementation (used in tests and local dev).

Production target is Postgres; the SQL here sticks to the portable subset so the same
statements work with a `?`→`%s` paramstyle shim (see `Store.q`).
"""
from __future__ import annotations

import json
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from ..core.hashing import sha
from ..core.timeutil import utcnow


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    # -- infra -----------------------------------------------------------------
    def migrate(self) -> None:
        sql = resources.files("nba_platform.db").joinpath("schema.sql").read_text()
        self.conn.executescript(sql)

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)))

    def x(self, sql: str, params: Iterable[Any] = ()) -> None:
        self.conn.execute(sql, tuple(params))

    def commit(self) -> None:
        self.conn.commit()

    # -- raw payload archive ----------------------------------------------------
    def archive_payload(self, source: str, endpoint: str, body: str | bytes,
                        fetched_at: str | None = None) -> str:
        text = body.decode("latin-1") if isinstance(body, bytes) else body
        digest = sha({"s": source, "e": endpoint, "b": text}, n=32)
        self.x("INSERT OR IGNORE INTO raw_payloads VALUES (?,?,?,?,?)",
               (digest, source, endpoint, fetched_at or utcnow().isoformat(), text))
        return digest

    # -- teams / games ------------------------------------------------------------
    def upsert_team(self, team_id: int, tricode: str, city: str | None, name: str | None) -> None:
        self.x("""INSERT INTO teams VALUES (?,?,?,?)
                  ON CONFLICT(team_id) DO UPDATE SET tricode=excluded.tricode,
                  city=excluded.city, name=excluded.name""", (team_id, tricode, city, name))

    def upsert_game(self, g: dict) -> None:
        self.x("""INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(game_id) DO UPDATE SET tip_utc=excluded.tip_utc,
                    game_date_et=excluded.game_date_et, status=excluded.status,
                    home_score=excluded.home_score, away_score=excluded.away_score,
                    periods=excluded.periods, home_poss=COALESCE(excluded.home_poss, home_poss),
                    away_poss=COALESCE(excluded.away_poss, away_poss), flags=COALESCE(excluded.flags, flags),
                    updated_at=excluded.updated_at""",
               (g["game_id"], g["season"], g["game_date_et"], g["tip_utc"], g["home_team_id"],
                g["away_team_id"], g.get("status", 1), g.get("home_score"), g.get("away_score"),
                g.get("periods"), g.get("home_poss"), g.get("away_poss"), g.get("flags"), utcnow().isoformat()))

    def games_on(self, game_date_et: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM games WHERE game_date_et=? ORDER BY tip_utc", (game_date_et,))

    # -- injuries -------------------------------------------------------------------
    def insert_injury_obs(self, o: dict) -> None:
        self.x("""INSERT INTO injury_status(player_id, team_id, game_id, status, origin, reason,
                    source, report_ts, observed_at, minutes_cap, confidence)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
               (o["player_id"], o.get("team_id"), o.get("game_id"), o["status"], o["origin"],
                o.get("reason"), o["source"], o.get("report_ts"), o["observed_at"],
                o.get("minutes_cap"), o.get("confidence")))

    def injury_state_asof(self, game_id: str, as_of: str) -> dict[str, sqlite3.Row]:
        """Latest observation per player for a game, using only rows observed <= as_of.

        Official origin wins over licensed feeds when both exist at the latest timestamp.
        """
        rows = self.q("""SELECT * FROM injury_status WHERE game_id=? AND observed_at<=?
                         ORDER BY observed_at,
                           CASE origin WHEN 'MODEL_INFERENCE' THEN 0 WHEN 'LICENSED_FEED' THEN 1
                                       ELSE 2 END, obs_id""", (game_id, as_of))
        state: dict[str, sqlite3.Row] = {}
        for r in rows:
            state[r["player_id"]] = r
        return state

    # -- lineups ------------------------------------------------------------------------
    def insert_lineup(self, game_id: str, team_id: int, state: str, starters: list[str],
                      source: str, observed_at: str) -> None:
        self.x("INSERT OR REPLACE INTO lineup_snapshots VALUES (?,?,?,?,?,?)",
               (game_id, team_id, state, json.dumps(starters), source, observed_at))

    def lineup_asof(self, game_id: str, team_id: int, as_of: str) -> sqlite3.Row | None:
        rows = self.q("""SELECT * FROM lineup_snapshots WHERE game_id=? AND team_id=? AND observed_at<=?
                         ORDER BY CASE state WHEN 'CONFIRMED' THEN 1 ELSE 0 END, observed_at""",
                      (game_id, team_id, as_of))
        return rows[-1] if rows else None

    # -- logs ------------------------------------------------------------------------------
    def upsert_log(self, r: dict) -> None:
        cols = ["game_id", "player_id", "team_id", "game_date_et", "started", "played", "minutes",
                "not_playing_reason", "pts", "reb", "ast", "fg3m", "fg3a", "fga", "fgm", "fta",
                "ftm", "oreb", "dreb", "tov", "pf", "plus_minus", "stl", "blk", "rim_fga", "rim_fgm", "mid_fga",
                "c3_fga", "ab3_fga", "ft_trips", "team_poss_on", "team_fga_on", "team_fgm_on", "team_miss_on",
                "opp_fga_on", "opp_miss_on", "team_pts_on", "opp_pts_on"]
        self.x(f"INSERT OR REPLACE INTO player_game_logs({','.join(cols)}) "
               f"VALUES ({','.join('?' * len(cols))})", [r.get(c) for c in cols])

    def logs_before(self, game_date_et: str) -> list[sqlite3.Row]:
        """Strictly earlier game dates only: the leakage guard for training."""
        return self.q("SELECT * FROM player_game_logs WHERE game_date_et < ? ORDER BY game_date_et",
                      (game_date_et,))

    # -- forecasts -----------------------------------------------------------------------------
    def insert_forecast(self, snap: dict, minutes_rows: list[dict]) -> None:
        self.x("""INSERT INTO forecast_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
               (snap["snapshot_id"], snap["game_id"], snap["refresh_window"], snap["created_at"],
                snap["model_name"], snap["model_version"], snap["input_hash"],
                json.dumps(snap["hashes"]), snap["flag"], json.dumps(snap["context"]),
                snap["seed"], snap["n_sims"]))
        for m in minutes_rows:
            self.x("""INSERT INTO minutes_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (snap["snapshot_id"], m["player_id"], m["team_id"], m.get("scenario", "BASE"),
                    m["mean"], m["median"], m["p10"], m["p25"], m["p75"], m["p90"], m["p_dnp"],
                    m.get("role_tier")))

    def supersede(self, old_id: str, new_id: str) -> None:
        self.x("UPDATE forecast_snapshots SET superseded_by=? WHERE snapshot_id=?", (new_id, old_id))

    def latest_forecast(self, game_id: str) -> sqlite3.Row | None:
        rows = self.q("""SELECT * FROM forecast_snapshots WHERE game_id=? AND superseded_by IS NULL
                         ORDER BY created_at DESC LIMIT 1""", (game_id,))
        return rows[0] if rows else None

    # -- jobs ---------------------------------------------------------------------------------------
    def log_job(self, job_id: str, job: str, status: str, detail: str = "",
                started_at: str | None = None) -> None:
        now = utcnow().isoformat()
        self.x("""INSERT INTO job_runs VALUES (?,?,?,?,?,?)
                  ON CONFLICT(job_id) DO UPDATE SET finished_at=excluded.finished_at,
                  status=excluded.status, detail=excluded.detail""",
               (job_id, job, started_at or now, now if status != "RUNNING" else None, status, detail))
