# NBA Projections Platform

Plataforma de proyecciones NBA con modelo propio (Champion `STACK_V2_ENV`: minutos, oportunidades, Monte Carlo
conjunto de 10,000 simulaciones, mercados de juego, calibración `BETA_MKT`), snapshots inmutables antes de cada
partido, liquidación automática y página web.

## Operación en la nube (sin computadora personal)

- `.github/workflows/nba-cycle.yml` corre cada 30 min en GitHub Actions: descarga datos públicos de nba.com,
  ejecuta el ciclo del modelo, liquida partidos terminados, valida la exportación y publica `web/` en GitHub Pages.
- Rama `main`: código y página. Rama `state`: datos persistentes (archivo de partidos, bases de snapshots),
  un solo commit reemplazado en cada ejecución.
- La página muestra el estado real de cada ejecución (fuentes, pasos, errores).

# NBA Projection & Market Intelligence Platform ($0 budget)

Data → rosters → injuries → lineups → **minutes distributions**, with immutable pregame snapshots
and a dependency-aware daily autopilot. See `docs/DIAGNOSTICO.md` for the full technical diagnosis.

```
pip install -e ".[test]"
pytest -q                                    # 34 tests, ~6 s, no network
nba-platform init  --db nba.db
nba-platform daily --db nba.db               # schedule + yesterday's finals (needs network)
nba-platform tick  --db nba.db               # injuries, pregame starters, refresh windows
nba-platform backtest --db nba.db --from 2026-11-01 --to 2026-11-12
```

| Module | Spec name |
|---|---|
| `roster/sync.py` | NBA_ROSTER_SYNC_V1 |
| `injuries/monitor.py`, `injuries/play_rates.py` | NBA_INJURY_MONITOR_V1 |
| `lineups/resolve.py` | Active lineup resolution, availability scenarios |
| `minutes/engine.py`, `minutes/forecast.py` | NBA_MINUTES_ENGINE_V1 |
| `minutes/validate.py` | Walk-forward validation, champion/challenger compare |
| `pipeline/autopilot.py`, `pipeline/ingest.py` | NBA_DAILY_AUTOPILOT_V1 |
| `core/hashing.py` | Hash tree + recompute plan |
| `db/schema.sql` | Schema (immutable forecast snapshots enforced by triggers) |

`tests/synthetic.py` is a **test-only** synthetic league with planted structure. It is never
loaded into a production database and its numbers are not NBA results.

## Phase 2 on real data ($0)

```
python scripts/build_local_db.py                     # open PBP dataset -> data/nba.db (SQLite)
python scripts/backtest_real.py 2025-11-01 2026-04-12 EXPECTED
python scripts/challenger_minutes.py CHALLENGER_B 2025-11-01 2026-04-12 1
python scripts/validate_sim_real.py 2025-11-15 2026-04-12 3 rating
```

| Module | Spec name |
|---|---|
| `sources/pbp_reconstruct.py`, `sources/open_dataset.py` | free historical data: minutes, starters, stints, possessions, box, on-court context |
| `sim/rates.py` | NBA_OPPORTUNITY_RATES_V1 / efficiency |
| `sim/game_sim.py` | NBA_FULL_PLAYER_MONTE_CARLO_V1 (joint, calibrated, accounting-checked) |
| `sim/markets.py` | NBA_PLAYER_PROPS_V1, NBA_GAME_MARKETS_V1, Phase-4 book comparison hook |
| `sim/calibration.py` | NBA_PROB_CALIBRATION_V1 (shadow) |
| `sim/validate.py` | walk-forward props + markets validation |
| `models/registry.json` | champion / challenger history with real metrics |

Results and limitations: `docs/FASE2_DATOS_REALES.md`.

## Model Improvement V2 (current champion: STACK_V2_ENV)

```
python scripts/build_real_dataset.py            # leak-free absences (v2)
python scripts/build_2026_27.py                 # 2026-27 from captured CDN JSONL, validated vs official box
python scripts/build_v3_with_2026_27.py         # v3 = v2 + 2026-27 with official availability
python scripts/minutes_v2.py M_DNP 2025-11-01 2026-04-12 2
python scripts/rating_v2.py 2025-11-01 2026-04-12 1500
python scripts/opportunity_v2.py 2025-11-01 2026-04-12 HL45
python scripts/validate_stack_v2.py STACK_V2_ENV 2026-10-27 2026-11-12 1 v3 '{"dnp_model":"competition"}'
python scripts/compare_stack.py v3_2026-10-27_2026-11-12_1 CHAMPION STACK_V2_ENV
scripts/daily_mac.sh                            # $0 daily run on the Mac (connectivity check first)
```

New modules: `rating/team_rating.py` (NBA_TEAM_RATING_V2), `sources/cdn_jsonl.py`, `pipeline/capture.py`.
Report: `docs/FASE3_MODEL_V2.md`. All decisions with CIs: `models/registry.json`.
