# NBA Projection & Market Intelligence Platform — Diagnóstico técnico y Fase 1

Fecha: 13-nov-2026 · Estado: Fase 1 implementada y probada **offline**; ninguna fuente en vivo ha sido consultada todavía desde el código (ver §0).

---

## 0. Lo que es verdad hoy (léelo primero)

| Afirmación | Estado |
|---|---|
| Código de Fase 1 (datos, rosters, lesiones, alineaciones, minutos, pipeline) | Escrito, 34 tests pasando |
| Parsers de feeds reales (CDN NBA, PDF oficial de lesiones, balldontlie) | Probados con fixtures que siguen la **estructura documentada**; **no** ejecutados contra los endpoints reales |
| Motor de minutos | Validado sobre una **liga sintética** con verdad conocida. **No** hay métricas sobre datos NBA reales todavía |
| Proyecciones de puntos/rebotes/asistencias/triples, Monte Carlo completo, mercados, UI | **No construidos** (Fases 2-5) |

El sandbox donde se construyó bloquea la salida de red hacia `cdn.nba.com`, `stats.nba.com` y `ak-static.cms.nba.com` (verificado: el proxy rechaza la conexión). Por eso el primer despliegue en la nube debe vigilarse: los parsers lanzan `SchemaDrift` en vez de producir datos incorrectos si la estructura real difiere.

---

## 1. Diagrama de arquitectura

```
                 ┌──────────────── FUENTES ────────────────┐
 cdn.nba.com (gratis)        PDF oficial de lesiones       balldontlie GOAT ($39.99/mes)   The Odds API (Fase 4)
 schedule · boxscore · pbp   ak-static.cms.nba.com          activos · lesiones · lineups    líneas + histórico
        │                           │                              │                            │
        ▼                           ▼                              ▼                            ▼
 ┌──────────────────────────────── INGESTA (idempotente, payload crudo archivado por hash) ───────────────┐
 └───────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                │                   │                   │                                │
  ROSTER_SYNC_V1   INJURY_MONITOR_V1   LINEUP RESOLUTION   PLAYER_GAME_LOGS                   MARKET_LINES
  (stints con      (append-only,       EXPECTED/CONFIRMED  (verdad post-partido)              (nunca entra
   fechas)          OFFICIAL vs         + escenarios                                           al modelo)
                    LICENSED)
        └──────────────┬─────────────────────┘                     │                                │
                       ▼                                           │                                │
              ROTATION context  ◄──────────────────────────────────┘                                │
                       ▼                                                                            │
              MINUTES_ENGINE_V1  (distribuciones, Σ=240+25·OT)            ← FASE 1 TERMINA AQUÍ      │
                       ▼                                                                            │
   F2: POSSESSIONS → USAGE → SHOT ATTEMPTS → SCORING/REB/AST/3PM → FULL MONTE CARLO (posesión a posesión)
                       ▼                                                                            │
   F3: GAME MARKETS (misma simulación conjunta)                                                     │
                       ▼                                                                            ▼
   F4: MARKET VALUE ENGINE (prob. propia vs no-vig, EV, EV conservador)  ◄─────────────────── líneas
                       ▼
   SNAPSHOTS INMUTABLES → API PÚBLICA (0 simulaciones por request) → UI
                       ▼
   F5: SETTLEMENT → VALIDACIÓN → CHAMPION / CHALLENGERS (shadow) → reportes diario/semanal
```

Cada capa tiene su hash (`ROSTER, INJURY, STARTERS, MODEL → ROTATION → MINUTES → OPPORTUNITY → DISTRIBUTION`; `MARKET` es hoja aparte). Un cambio de cuota sólo recalcula `MARKET_COMPARISON`; un cambio de lesión recalcula todo lo que cuelga de `INJURY`. Probado en `test_hash_tree_dependency_aware_plan`.

---

## 2. Auditoría de fuentes de datos

| Fuente | Qué da | Costo | Acceso desde nube | Licencia / riesgo | Uso propuesto |
|---|---|---|---|---|---|
| **cdn.nba.com** (JSON de nba.com) | Calendario de temporada, scoreboard, boxscore por partido (titulares, inactivos, minutos, stats), play-by-play (sustituciones) | Gratis | Reportado como accesible (a diferencia de stats.nba.com) | No documentado, uso no licenciado formalmente; puede cambiar sin aviso | Columna vertebral gratuita de V1 |
| **stats.nba.com** (vía `nba_api`) | Tracking: touches, drives, potential assists, rebound chances, shot zones | Gratis | **Bloqueado** desde IPs de AWS/GCP/Azure + fingerprinting TLS de Akamai | Igual que arriba | Sólo con proxy residencial o una máquina no-datacenter; necesario para Fase 2 (oportunidades) |
| **PDF oficial de lesiones** | Estado oficial (Available/Probable/Questionable/Doubtful/Out) + razón, cada 15 min en días de juego | Gratis | No probado desde nube aún | Publicación oficial de la liga | Fuente OFFICIAL del injury monitor |
| **balldontlie** | Activos con equipo, lesiones, lineups, box scores, props | Free / $9.99 / **$39.99 (GOAT)** al mes | Sí (API con key) | Licencia comercial del proveedor | Respaldo para rosters/lesiones; lineups |
| **SportsDataIO** | Lineups proyectados y confirmados, depth charts, lesiones, headshots | Cotización (enterprise) | Sí | Licencia clara | Upgrade cuando haya presupuesto: mejora la latencia de titulares confirmados |
| **Sportradar** | Feed oficial completo | Enterprise | Sí | Oficial | Fuera de alcance V1 |
| **The Odds API** | Líneas de juego y player props, histórico en planes pagados | Free 500 créditos; $30-$249/mes | Sí | Licencia comercial | Fase 4 (comparación con sportsbooks) |

Recomendación V1: cdn.nba.com + PDF oficial (gratis) + balldontlie GOAT ($39.99) ≈ **$40/mes**, más The Odds API (~$30-$119) cuando empiece la Fase 4. Fase 2 necesita además un camino estable a datos de tracking (proxy residencial ~$10-$50/mes, o alternativa licenciada).

---

## 3. Propuesta de base de datos

Implementada en `nba_platform/db/schema.sql` (SQLite en tests; Postgres en producción). Tablas clave:

- `players` + `player_id_xref`: id interno estable `P<personId NBA>`; ids de otras fuentes mapeados, nunca duplicados.
- `roster_stints`: pertenencia a equipo con `start_date/end_date`; nunca se borra, sólo se cierra.
- `transactions`: log idempotente (hash de contenido).
- `injury_status`: **append-only** con `observed_at` (cuándo lo vimos) y `origin` (OFFICIAL / LICENSED_FEED / MODEL_INFERENCE). Consultas "as-of" para reconstrucción sin fuga.
- `injury_report_coverage`: qué equipos ya habían enviado su reporte oficial.
- `lineup_snapshots`: EXPECTED / CONFIRMED con timestamp.
- `player_game_logs`: verdad post-partido.
- `raw_payloads`: cada respuesta externa archivada por hash.
- `forecast_snapshots` + `minutes_forecasts`: **inmutables por trigger** (no UPDATE, no DELETE; sólo `superseded_by`).
- `market_lines`: apertura / intermedias / cierre con `closing_confirmed` (Fase 4).
- `model_registry`, `validation_runs`, `job_runs`.

---

## 4. Plan de datos actuales vs históricos

- **Actual manda**: el roster de hoy sale de `roster_stints` vigente; el estado de cada jugador sale del último reporte oficial *observado*; los titulares, del último lineup CONFIRMED (si no hay, EXPECTED del modelo).
- **Histórico estima habilidades**: logs con ponderación por recencia (vida media 12 partidos del equipo; temporada anterior ×0.35).
- **Reconstrucción histórica**: con el archivo `raw_payloads` + `injury_status.observed_at` podemos reconstruir el estado de información en cualquier minuto antes del tip. Para temporadas pasadas sólo podemos reconstruir lo que exista archivado: los PDFs oficiales históricos siguen en la URL de la liga (rastreables), pero los lineups "esperados" históricos y las líneas históricas requieren proveedor pagado.
- **Riesgos de fuga identificados y controlados**:
  1. Entrenar con partidos del mismo día → `logs_before()` usa `<` estricto y el motor lanza error si recibe datos ≥ fecha.
  2. Usar el estado de lesión final en vez del pregame → todas las consultas son as-of `observed_at`.
  3. Inactivos del boxscore final usados como "conocidos pregame" en backtest → aceptable sólo como proxy de la ventana T-30m (la lista de inactivos es oficial antes del tip); documentado.
  4. Margen final real usado en el entrenamiento del efecto blowout → es correcto para *aprender* la conducta del coach; en la predicción el margen se **simula**, nunca se observa.
  5. Líneas de sportsbook → no existen como input en ninguna capa del modelo (el hash `MARKET` no es padre de nada del modelo).

---

## 5. Motor de lesiones (NBA_INJURY_MONITOR_V1)

- Parser posicional del PDF oficial (reconstruye filas por coordenadas x/y, resuelve razones partidas arriba/abajo de la fila, arrastra fecha/hora/matchup/equipo) + parser de texto de respaldo.
- Estado oficial manda; la razón sólo refina OUT en REST / SUSPENDED / NOT_WITH_TEAM (G League, two-way).
- Nombres que no coinciden con exactamente un jugador del roster → `unresolved`, **no se guardan**.
- Restricciones de minutos sólo si la fuente las declara.
- `diff()` clasifica cambios HIGH/MEDIUM/LOW (¿cruza la frontera juega/no juega? ¿es jugador CORE?).
- Probabilidad de jugar para QUESTIONABLE/DOUBTFUL: **sólo** desde nuestro propio archivo (`fit_status_play_rates`: estado oficial a T-90m vs si jugó), con mínimo 40 observaciones; si no, escenarios separados sin probabilidad.

## 6. Modelo de minutos (NBA_MINUTES_ENGINE_V1)

Regresión bayesiana ridge ponderada por recencia, por jugador y en su equipo actual, con **medias a priori informativas**:

```
min | juega = b_titular·S + b_banca·(1−S)
            + BX·(s_titular·S + s_banca·(1−S))      BX = max(0, |margen| − 12)
            + Σ_j d_ij · ausente_j
            + ruido(σ_i)
```

- `b_*`: historial propio del jugador (cualquier equipo) encogido hacia niveles de liga.
- `s_*`: **conducta del coach** ante blowouts, estimada dentro-de-jugador por equipo, encogida a liga.
- `d_ij`: quién absorbe los minutos de quién. Prior estructural (minutos de j × similitud posicional × espacio disponible), que los partidos reales con/sin j sobreescriben. Ausencias nunca vistas usan el prior y se marcan `UNSEEN_ABSENCE`.
- P(DNP) sólo para banca, por contexto de ausencias.
- Titulares esperados: tasa de titularidad condicionada a partidos donde faltaban los mismos titulares; si esa ausencia nunca se observó, respaldo posicional.
- Traspasos: σ × (1 + 1/(1+n partidos con el equipo nuevo)) → `NEW_TEAM`. No hereda el uso del equipo anterior.
- Regreso de lesión: rampa y ensanchamiento aprendidos de eventos históricos de liga (≥10 eventos); si no hay suficientes → factor 1.25 marcado `UNCALIBRATED`.
- Simulación: margen → BX y prórrogas → minutos crudos → DNP → **proyección exacta** a Σ = 240 + 25·OT, 0 ≤ m ≤ 48+5·OT (o restricción), ajustando a cada jugador en proporción a su varianza (condicionamiento gaussiano sobre la suma). Jugadores OUT excluidos antes de simular. Semillas reproducibles derivadas de (game_id, input hash).
- Salidas: media, mediana, P10/P25/P75/P90, P(DNP), tier CORE/ROTATION/FRINGE, flags.

**Limitación explícita**: el margen de Fase 1 es un proxy (ratings de diferencial de puntos ponderado + ventaja local, todo estimado de datos). Foul trouble no está modelado explícitamente (lo absorbe σ); llega en Fase 2 con la simulación por posesiones.

### Resultados de validación (LIGA SINTÉTICA, no NBA)

Walk-forward, 8 fechas, 593 filas jugador-partido, 24 partidos; baseline = media/cuantiles de los últimos 10 partidos jugados.

| | MAE | CRPS | Cobertura P10-P90 (nominal 0.80) | Cobertura P25-P75 (0.50) |
|---|---|---|---|---|
| Baseline últimos 10 | 3.66 | 2.72 | 0.55 | 0.35 |
| Motor (titulares EXPECTED) | 3.47 | 2.35 | 0.82 | 0.53 |

Comparación pareada con bootstrap por partido: CRPS −0.37 (IC95 −0.51 a −0.24) → **PROMOTE** con la regla CRPS-primaria; MAE −0.19 (IC95 −0.38 a +0.01) → **no significativo** por sí solo. Titulares: cobertura 0.73 (sub-dispersión) — a vigilar con datos reales. La liga sintética es más fácil que la NBA real (titulares deterministas por profundidad); estas cifras prueban que el motor recupera la estructura plantada, **no** su precisión en producción.

## 7. Modelo de rotación

Fase 1: rotación implícita (quién juega y cuánto, según disponibilidad y titulares). Fase 2: modelo explícito de *stints* a partir de las sustituciones del play-by-play del CDN (quién entra por quién, en qué minuto, por cuarto) → combinaciones de quintetos y minutos por quinteto, que alimentan el usage dependiente de compañeros.

## 8. Modelos de oportunidad (Fase 2)

Por posesión simulada: quinteto en cancha → reparto de usage (según roles y compañeros presentes) → tipo de tiro (aro / media / triple C&S / triple pull-up / TL) → conversión (prior jerárquico por zona, encogido por muestra) → rebote (oportunidades = tiros fallados elegibles, reparto por posición y tamaño del quinteto) → asistencia (sólo sobre canastas de compañeros). Requiere tracking de stats.nba.com (ver §2) o un proveedor licenciado; sin eso se usa la versión boxscore (FGA/3PA/FTA/OREB/DREB) con menor resolución.

## 9. Arquitectura Monte Carlo (Fase 2)

Simulación posesión a posesión por partido, 10.000 sims (100.000 tras pruebas de convergencia), semilla reproducible, mismo estado de juego para todas las estadísticas; contabilidad de equipo verificada en cada sim (Σ puntos jugadores = puntos equipo; asistencias ≤ canastas de compañeros; rebotes ≤ fallos elegibles; 5 en cancha).

## 10. Game markets (Fase 3)

Marcador conjunto, moneyline, spread, total, team totals y alternativas leídos de **la misma** simulación. Sustituye el proxy de margen de Fase 1.

## 11. Comparación con mercado (Fase 4)

Después de existir las distribuciones propias: P(over/under/push) con reglas de liquidación, cuota justa, probabilidad implícita y no-vig, gap, EV y EV conservador (con incertidumbre del modelo). Etiquetas: NO BET / MODEL LEAN / POSITIVE EV / INSUFFICIENT DATA. Nunca LOCK.

## 12. Scheduler diario

- `daily` (~09:00 ET): calendario + finales de ayer (logs, marcadores, rosters por boxscore).
- `tick` (cada 5 min, 11:00–01:00 ET): reporte oficial de lesiones más reciente; boxscore pregame en los últimos 90 min (titulares CONFIRMED + inactivos); ventanas INITIAL / 6H / 90M / 30M / 15M; sólo recalcula si cambió un hash aguas arriba de MINUTES; las ventanas perdidas se saltan.
- Infra recomendada: Cloud Run Jobs + Cloud Scheduler (o Fly.io) + Postgres gestionado (Neon/Supabase) + alertas de jobs fallidos. Dockerfile incluido. GitHub Actions sólo para CI (su cron no es fiable a 5 min).

## 13. Marco de validación

Walk-forward estricto; métricas MAE, sesgo, RMSE, CRPS, coberturas; desgloses por titular/banca, tier y contexto de lesión. Fase 5 añade descomposición de error (minutos vs usage vs eficiencia vs blowout vs varianza) y cohortes `NBA_DAILY_COHORT_###`.

## 14. Champion / Challenger

`compare()` hace comparación pareada sobre las mismas filas con bootstrap por partido. PROMOTE sólo si el IC95 de la métrica primaria (CRPS) está entero bajo cero, la secundaria no empeora significativamente y la cobertura no se aleja del nominal. Probado: un modelo contra sí mismo nunca se promueve. Sin auto-promoción en producción; `model_registry` guarda hipótesis y roles.

## 15. Fases de implementación

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Datos, rosters, lesiones, alineaciones, minutos, pipeline, snapshots | **Hecha (offline)** |
| 1b | Primer despliegue en nube; correr `daily` + `backtest` sobre la temporada 2025-26 y lo jugado de 2026-27; calibrar σ de titulares y rampas | Siguiente |
| 2 | Stints/rotación, posesiones, usage, tiros, puntos/reb/ast/3PM, Monte Carlo por posesión | |
| 3 | Game markets desde la simulación conjunta | |
| 4 | Odds, line history, valor de mercado | |
| 5 | Settlement, track record, cohortes, reportes, challengers en shadow | |
| 6 | UI pública (Overview, Props, Game Markets, Results, Player pages) + Admin | |

Decisión pendiente tuya antes de 1b: presupuesto de datos (balldontlie $40/mes como mínimo; ¿proxy para tracking en Fase 2?) y dónde alojar (GCP vs Fly.io).
