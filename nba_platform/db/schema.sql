-- NBA platform schema v1. Portable subset of SQL (runs on SQLite for tests; on Postgres in prod,
-- swap TEXT-JSON columns for JSONB and the SQLite triggers for the Postgres equivalents in
-- schema_postgres_notes.md). All timestamps are ISO-8601 UTC strings.

CREATE TABLE IF NOT EXISTS teams (
    team_id      INTEGER PRIMARY KEY,         -- NBA teamId (e.g. 1610612747)
    tricode      TEXT NOT NULL UNIQUE,
    city         TEXT,
    name         TEXT
);

-- Internal stable player id = 'P' || nba personId. External ids live in player_id_xref.
CREATE TABLE IF NOT EXISTS players (
    player_id      TEXT PRIMARY KEY,
    nba_person_id  INTEGER UNIQUE,
    full_name      TEXT NOT NULL,
    first_name     TEXT,
    last_name      TEXT,
    name_key       TEXT NOT NULL,             -- normalised for matching ("lastname|firstname")
    position       TEXT,
    birthdate      TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_players_name_key ON players(name_key);

CREATE TABLE IF NOT EXISTS player_id_xref (
    source     TEXT NOT NULL,                 -- 'nba', 'balldontlie', 'odds_api', 'sportsdataio'
    source_id  TEXT NOT NULL,
    player_id  TEXT NOT NULL REFERENCES players(player_id),
    PRIMARY KEY (source, source_id)
);

-- Effective-dated team membership. end_date NULL = current. Never deleted, only closed.
CREATE TABLE IF NOT EXISTS roster_stints (
    player_id      TEXT NOT NULL REFERENCES players(player_id),
    team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    contract_type  TEXT NOT NULL DEFAULT 'STANDARD',   -- STANDARD | TWO_WAY | TEN_DAY | EXHIBIT_10
    start_date     TEXT NOT NULL,
    end_date       TEXT,
    source         TEXT NOT NULL,
    recorded_at    TEXT NOT NULL,
    PRIMARY KEY (player_id, team_id, start_date)
);
CREATE INDEX IF NOT EXISTS ix_stints_team ON roster_stints(team_id, end_date);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id          TEXT PRIMARY KEY,         -- content hash => idempotent inserts
    player_id       TEXT NOT NULL,
    txn_type        TEXT NOT NULL,
    from_team_id    INTEGER,
    to_team_id      INTEGER,
    effective_date  TEXT NOT NULL,
    source          TEXT NOT NULL,
    detail          TEXT,
    detected_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,           -- NBA gameId '0022600123'
    season        TEXT NOT NULL,
    game_date_et  TEXT NOT NULL,
    tip_utc       TEXT NOT NULL,
    home_team_id  INTEGER NOT NULL,
    away_team_id  INTEGER NOT NULL,
    status        INTEGER NOT NULL DEFAULT 1, -- 1 scheduled, 2 live, 3 final
    home_score    INTEGER,
    away_score    INTEGER,
    periods       INTEGER,
    home_poss     INTEGER,
    away_poss     INTEGER,
    flags         TEXT,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(game_date_et);

-- Every external response is archived by content hash: enables leakage-free reconstruction.
CREATE TABLE IF NOT EXISTS raw_payloads (
    content_sha  TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    endpoint     TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    body         TEXT NOT NULL
);

-- Append-only availability observations. Latest per (player, game) as-of T = state at T.
CREATE TABLE IF NOT EXISTS injury_status (
    obs_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id     TEXT NOT NULL,
    team_id       INTEGER,
    game_id       TEXT,
    status        TEXT NOT NULL,
    origin        TEXT NOT NULL,              -- OFFICIAL | LICENSED_FEED | MODEL_INFERENCE
    reason        TEXT,
    source        TEXT NOT NULL,
    report_ts     TEXT,                       -- as-of time stated by the source
    observed_at   TEXT NOT NULL,              -- when WE saw it (leakage guard uses this)
    minutes_cap   REAL,                       -- only if explicitly reported
    confidence    REAL                        -- NULL for official statuses
);
CREATE INDEX IF NOT EXISTS ix_injury_pg ON injury_status(player_id, game_id, observed_at);

-- Whether each team had submitted its official injury report entries, per report we ingested.
CREATE TABLE IF NOT EXISTS injury_report_coverage (
    game_id      TEXT NOT NULL,
    team_id      INTEGER NOT NULL,
    submitted    INTEGER NOT NULL,
    report_ts    TEXT,
    observed_at  TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, observed_at)
);

CREATE TABLE IF NOT EXISTS lineup_snapshots (
    game_id      TEXT NOT NULL,
    team_id      INTEGER NOT NULL,
    state        TEXT NOT NULL,               -- EXPECTED | CONFIRMED
    starters     TEXT NOT NULL,               -- JSON array of player_id
    source       TEXT NOT NULL,
    observed_at  TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, observed_at)
);

CREATE TABLE IF NOT EXISTS player_game_logs (
    game_id     TEXT NOT NULL,
    player_id   TEXT NOT NULL,
    team_id     INTEGER NOT NULL,
    game_date_et TEXT NOT NULL,
    started     INTEGER NOT NULL,
    played      INTEGER NOT NULL,
    minutes     REAL NOT NULL,
    not_playing_reason TEXT,
    pts INTEGER, reb INTEGER, ast INTEGER, fg3m INTEGER, fg3a INTEGER,
    fga INTEGER, fgm INTEGER, fta INTEGER, ftm INTEGER, oreb INTEGER, dreb INTEGER,
    tov INTEGER, pf INTEGER, plus_minus REAL,
    stl INTEGER, blk INTEGER,
    -- derived from play-by-play (shot zones + on-court context = free opportunity estimators)
    rim_fga INTEGER, rim_fgm INTEGER, mid_fga INTEGER, c3_fga INTEGER, ab3_fga INTEGER, ft_trips INTEGER,
    team_poss_on REAL, team_fga_on REAL, team_fgm_on REAL, team_miss_on REAL,
    opp_fga_on REAL, opp_miss_on REAL, team_pts_on REAL, opp_pts_on REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_logs_player ON player_game_logs(player_id, game_date_et);

-- Immutable pregame forecasts (spec §45). Only superseded_by may be set after insert.
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    snapshot_id    TEXT PRIMARY KEY,
    game_id        TEXT NOT NULL,
    refresh_window TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    input_hash     TEXT NOT NULL,
    hashes         TEXT NOT NULL,             -- JSON hash tree
    flag           TEXT NOT NULL,             -- OK | PARTIAL_DATA | STALE | UNAVAILABLE
    context        TEXT NOT NULL,             -- JSON injury / lineup context used
    seed           INTEGER NOT NULL,
    n_sims         INTEGER NOT NULL,
    superseded_by  TEXT
);
CREATE INDEX IF NOT EXISTS ix_fs_game ON forecast_snapshots(game_id, created_at);

CREATE TABLE IF NOT EXISTS minutes_forecasts (
    snapshot_id  TEXT NOT NULL REFERENCES forecast_snapshots(snapshot_id),
    player_id    TEXT NOT NULL,
    team_id      INTEGER NOT NULL,
    scenario     TEXT NOT NULL DEFAULT 'BASE',
    mean REAL, median REAL, p10 REAL, p25 REAL, p75 REAL, p90 REAL,
    p_dnp        REAL,
    role_tier    TEXT,                        -- CORE | ROTATION | FRINGE
    PRIMARY KEY (snapshot_id, player_id, scenario)
);

-- Sportsbook line history (populated from Phase 3; never read by model layers).
CREATE TABLE IF NOT EXISTS market_lines (
    line_id      TEXT PRIMARY KEY,
    game_id      TEXT NOT NULL,
    player_id    TEXT,
    market       TEXT NOT NULL,
    book         TEXT NOT NULL,
    line         REAL,
    over_price   INTEGER,
    under_price  INTEGER,
    captured_at  TEXT NOT NULL,
    is_opening   INTEGER NOT NULL DEFAULT 0,
    closing_confirmed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    role          TEXT NOT NULL,              -- CHAMPION | CHALLENGER_A | CHALLENGER_B | CHALLENGER_C | RETIRED
    params_digest TEXT,
    hypothesis    TEXT,
    created_at    TEXT NOT NULL,
    promoted_at   TEXT,
    PRIMARY KEY (model_name, version)
);

CREATE TABLE IF NOT EXISTS validation_runs (
    run_id       TEXT PRIMARY KEY,
    model_name   TEXT NOT NULL,
    version      TEXT NOT NULL,
    cohort       TEXT,
    metrics      TEXT NOT NULL,               -- JSON
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_runs (
    job_id      TEXT PRIMARY KEY,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,                -- RUNNING | OK | FAILED | SKIPPED
    detail      TEXT
);

CREATE TRIGGER IF NOT EXISTS trg_forecast_immutable
BEFORE UPDATE ON forecast_snapshots
FOR EACH ROW WHEN
    NEW.snapshot_id IS NOT OLD.snapshot_id OR NEW.game_id IS NOT OLD.game_id OR
    NEW.created_at IS NOT OLD.created_at OR NEW.input_hash IS NOT OLD.input_hash OR
    NEW.hashes IS NOT OLD.hashes OR NEW.context IS NOT OLD.context OR
    NEW.model_version IS NOT OLD.model_version OR NEW.flag IS NOT OLD.flag
BEGIN
    SELECT RAISE(ABORT, 'forecast_snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_minutes_immutable
BEFORE UPDATE ON minutes_forecasts
BEGIN
    SELECT RAISE(ABORT, 'minutes_forecasts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_forecast_nodelete
BEFORE DELETE ON forecast_snapshots
BEGIN
    SELECT RAISE(ABORT, 'forecast_snapshots cannot be deleted');
END;

-- Phase 2/3: player prop distributions and game markets read from the joint simulation.
CREATE TABLE IF NOT EXISTS prop_forecasts (
    snapshot_id  TEXT NOT NULL REFERENCES forecast_snapshots(snapshot_id),
    player_id    TEXT NOT NULL,
    team_id      INTEGER NOT NULL,
    market       TEXT NOT NULL,              -- PTS REB AST 3PM PRA PR PA RA FGA 3PA FTA ...
    scenario     TEXT NOT NULL DEFAULT 'BASE',
    mean REAL, median REAL, p10 REAL, p25 REAL, p75 REAL, p90 REAL, sd REAL,
    minutes_mean REAL, p_dnp REAL,
    model_fair_line REAL, p_over_fair REAL, p_under_fair REAL,
    PRIMARY KEY (snapshot_id, player_id, market, scenario)
);

CREATE TABLE IF NOT EXISTS game_market_forecasts (
    snapshot_id  TEXT PRIMARY KEY REFERENCES forecast_snapshots(snapshot_id),
    payload      TEXT NOT NULL               -- JSON from sim.markets.game_markets + sim meta
);

CREATE TRIGGER IF NOT EXISTS trg_props_immutable BEFORE UPDATE ON prop_forecasts
BEGIN SELECT RAISE(ABORT, 'prop_forecasts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_gm_immutable BEFORE UPDATE ON game_market_forecasts
BEGIN SELECT RAISE(ABORT, 'game_market_forecasts are immutable'); END;

-- Prospective settlement (spec V3 §8): frozen pregame forecast vs official result. Immutable.
CREATE TABLE IF NOT EXISTS forecast_results (
    snapshot_id   TEXT NOT NULL REFERENCES forecast_snapshots(snapshot_id),
    game_id       TEXT NOT NULL,
    player_id     TEXT NOT NULL,              -- '' for game-level rows
    market        TEXT NOT NULL,              -- MIN PTS REB AST 3PM PRA | MARGIN TOTAL HOME_WIN
    scenario      TEXT NOT NULL,
    predicted_mean REAL, predicted_median REAL, p10 REAL, p90 REAL,
    p_over_fair_line REAL, fair_line REAL,
    actual        REAL,
    abs_error     REAL,
    in_p10_p90    INTEGER,
    settled_at    TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, player_id, market, scenario)
);
CREATE TRIGGER IF NOT EXISTS trg_results_immutable BEFORE UPDATE ON forecast_results
BEGIN SELECT RAISE(ABORT, 'forecast_results are immutable'); END;

-- ---------------------------------------------------------------------------------------------
-- V4: publication records. Which distribution was actually published for each component, with
-- the frozen scenario weights. Written in the same transaction as the snapshot (or, for
-- snapshots created before V4, reconstructed from the stored rows and marked backfilled=1).
CREATE TABLE IF NOT EXISTS forecast_publications (
    snapshot_id     TEXT NOT NULL REFERENCES forecast_snapshots(snapshot_id),
    component       TEXT NOT NULL,           -- MINUTES | PROPS | GAME
    team_id         INTEGER NOT NULL DEFAULT 0,  -- MINUTES: team; PROPS/GAME: 0 (game-level pair labels)
    publication     TEXT NOT NULL,           -- SINGLE | MIXTURE | CONDITIONAL
    published_label TEXT,                    -- stored label that IS the published distribution (NULL if CONDITIONAL)
    scenarios       TEXT NOT NULL,           -- JSON [{label, conditions:{player_id: PLAYS|OUT}, weight|null}]
    weights_source  TEXT,                    -- e.g. play_rates@<fit date> ; NULL when no weights
    weights_hash    TEXT,
    created_at      TEXT NOT NULL,
    backfilled      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (snapshot_id, component, team_id)
);
CREATE TRIGGER IF NOT EXISTS trg_pub_immutable BEFORE UPDATE ON forecast_publications
BEGIN SELECT RAISE(ABORT, 'forecast_publications are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_pub_nodelete BEFORE DELETE ON forecast_publications
BEGIN SELECT RAISE(ABORT, 'forecast_publications cannot be deleted'); END;

-- V4 settlement: one row per settled (snapshot, player, market) for the PUBLISHED distribution only.
CREATE TABLE IF NOT EXISTS settlements (
    snapshot_id     TEXT NOT NULL,
    game_id         TEXT NOT NULL,
    category        TEXT NOT NULL,           -- PROSPECTIVE | SHADOW | REPLAY (never mixed in reports)
    stack           TEXT,
    component       TEXT NOT NULL,           -- MINUTES | PROPS | GAME
    player_id       TEXT NOT NULL,           -- '' for game-level markets
    market          TEXT NOT NULL,
    publication     TEXT NOT NULL,
    settled_label   TEXT NOT NULL,
    settle_method   TEXT NOT NULL,           -- PUBLISHED_SINGLE | PUBLISHED_MIXTURE | CONDITIONAL_MATCHED
    weights_hash    TEXT,
    pred_mean REAL, pred_median REAL, p10 REAL, p90 REAL, p_over_fair REAL, fair_line REAL,
    actual REAL, abs_error REAL, in_p10_p90 INTEGER,
    settle_version  TEXT NOT NULL,
    settled_at      TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, component, player_id, market, settle_version)
);
CREATE TABLE IF NOT EXISTS settlement_events (
    snapshot_id TEXT NOT NULL, component TEXT NOT NULL, team_id INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,                    -- SETTLED | NO_MATCHING_SCENARIO | NO_PUBLICATION
    detail TEXT, settle_version TEXT NOT NULL, settled_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, component, team_id, settle_version)
);
CREATE TRIGGER IF NOT EXISTS trg_settle_immutable BEFORE UPDATE ON settlements
BEGIN SELECT RAISE(ABORT, 'settlements are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_settle_nodelete BEFORE DELETE ON settlements
BEGIN SELECT RAISE(ABORT, 'settlements cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS trg_sevents_immutable BEFORE UPDATE ON settlement_events
BEGIN SELECT RAISE(ABORT, 'settlement_events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_minutes_nodelete BEFORE DELETE ON minutes_forecasts
BEGIN SELECT RAISE(ABORT, 'minutes_forecasts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS trg_props_nodelete BEFORE DELETE ON prop_forecasts
BEGIN SELECT RAISE(ABORT, 'prop_forecasts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS trg_gm_nodelete BEFORE DELETE ON game_market_forecasts
BEGIN SELECT RAISE(ABORT, 'game_market_forecasts cannot be deleted'); END;

-- V6: the full Monte Carlo distribution of every published player market, stored at snapshot time
-- (exact integer counts over the n simulations; any line's P(over/under/push) is computed from it).
CREATE TABLE IF NOT EXISTS prop_distributions (
    snapshot_id TEXT NOT NULL REFERENCES forecast_snapshots(snapshot_id),
    player_id   TEXT NOT NULL,
    scenario    TEXT NOT NULL,
    market      TEXT NOT NULL,
    n           INTEGER NOT NULL,       -- number of simulations
    lo          INTEGER NOT NULL,       -- value of counts[0]
    counts      TEXT NOT NULL,          -- JSON list: counts[i] = #sims with value lo+i
    PRIMARY KEY (snapshot_id, player_id, scenario, market)
);
CREATE TRIGGER IF NOT EXISTS trg_pdist_immutable BEFORE UPDATE ON prop_distributions
BEGIN SELECT RAISE(ABORT, 'prop_distributions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_pdist_nodelete BEFORE DELETE ON prop_distributions
BEGIN SELECT RAISE(ABORT, 'prop_distributions cannot be deleted'); END;
