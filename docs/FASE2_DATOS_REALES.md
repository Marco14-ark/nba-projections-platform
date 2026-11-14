# NBA Platform — Presupuesto $0, datos reales y validación (13-nov-2026)

Regla vigente: **presupuesto total $0 MXN**. Nada de APIs, proxies, proveedores, bases de datos, hosting ni suscripciones de pago. Este documento reemplaza la recomendación de proveedores pagados de `DIAGNOSTICO_FASE1.md`; el resto de la Fase 1 se conserva.

---

## 1. Fuentes gratuitas: acceso verificado (no supuesto)

| Fuente | Prueba real desde el entorno de desarrollo | Cobertura | Condiciones | Uso |
|---|---|---|---|---|
| **github.com/shufinskiy/nba_data** (`cdnnba_<año>`) | ✅ Descargado vía `git` (partial clone). | Play-by-play completo del CDN NBA: **2024-25 (1,230 partidos)** y **2025-26 (1,230)** de temporada regular + 60 de playoffs 2026. **Sin archivo 2026-27** (repo actualizado al 6-nov-2026). | Apache-2.0 | Fuente histórica principal |
| cdn.nba.com (schedule, boxscore, pbp) | ❌ Bloqueado por el egress del entorno de desarrollo (no por la NBA). | Temporada en curso | Uso no licenciado formalmente; contenido de NBA.com | Fuente en vivo cuando corra en un runner con salida a internet (ver §6) |
| stats.nba.com / `nba_api` | ❌ Aquí; además NBA bloquea IPs de nube | Tracking (touches, potential assists, rebound chances) | Términos de NBA.com; la librería es gratis, el acceso no está garantizado | Solo desde IP residencial (tu Mac). No se usan proxies |
| PDF oficial de lesiones (ak-static.cms.nba.com) | ❌ Aquí (egress) | Estados oficiales cada 15 min | Publicación oficial | Parser listo y probado; corre en el runner |
| pypi / raw.githubusercontent / api.github | ❌ Aquí | — | — | Por eso no hay DuckDB/pyarrow: se usa SQLite + pandas (ya instalados) |
| Líneas de sportsbooks | No hay histórico gratuito verificable | — | — | **No se inventan cuotas.** Se publican distribuciones y "fair lines" del modelo, etiquetadas como tales |

Verificación externa de la reconstrucción: Luka Dončić 2025-26 según nuestra reconstrucción = 64 PJ, 33.5 pts, 7.7 reb, 8.3 ast; coincide con Land of Basketball (33.5 / 7.7 / 8.3 / 64).

## 2. Lo que se reconstruye gratis desde play-by-play (`sources/pbp_reconstruct.py`)

Por partido: minutos exactos por jugador (stints con reloj), titulares (lineup al inicio del P1), quintetos por stint, posesiones (cambio de posesión), boxscore completo (pts, FGA/FGM, 3PA/3PM, FTA/FTM, OREB/DREB, AST, TOV, PF, STL, BLK), zonas de tiro (aro / media / triple de esquina / arriba del arco) y **contexto en cancha** por jugador: posesiones del equipo, tiros fallados propios y rivales, canastas de compañeros.

Controles de calidad en cada partido: minutos del equipo = 240 + 25·OT, Σ puntos jugadores = marcador final, 5 titulares. **Resultado: 99.1 % de partidos pasan los tres; los 23 que fallan se marcan y se excluyen de la validación.**

Pace reconstruido: 100.5 (2024-25) y 101.1 (2025-26) posesiones por equipo por 48 min. Promedios de liga por equipo-partido: 114.7 pts, 89.2 FGA, 37.3 3PA, 26.7 AST.

### Estimadores en lugar de tracking (etiquetados como estimaciones)
| Variable de tracking (no gratuita) | Estimador observable | Validación |
|---|---|---|
| Rebound chances | REB / tiros fallados (propios o rivales) mientras está en cancha | Vía REB props (§4) |
| Potential assists | AST / canastas de compañeros mientras está en cancha | Vía AST props |
| Touches / usage | (FGA + TOV + viajes a la línea) / posesiones del equipo en cancha | Vía PTS props |
| Shot quality | Mezcla de zonas por jugador + % por zona con shrinkage + defensa rival por zona | Vía PTS/3PM |

**Limitación sin solución gratuita:** el play-by-play no trae la lista de inactivos. Un jugador que no entra no deja rastro. Se infiere: ausente con ≥12 min recientes → `INACTIVE_INFERRED`; si no, `DNP_INFERRED`. Son inferencias, no designaciones oficiales. En producción la lista real vendrá del PDF oficial y del boxscore pregame.

## 3. Motor de minutos con datos reales — backtest temporal

Walk-forward 1-nov-2025 → 12-abr-2026, **153 fechas, 31,030 jugador-partido**, reentrenando cada día solo con partidos anteriores; titulares **EXPECTED** (predichos, sin fuga).

| Modelo | MAE | CRPS | Cobertura P10-P90 (nominal 0.80) |
|---|---|---|---|
| Baseline: últimos 10 jugados | 5.61 | 4.13 | 0.54 |
| Baseline fuerte: últimos 10 disponibles (DNP = 0) | **4.64** | 3.41 | 0.70 |
| Motor 1.0.0 (Fase 1) | 5.13 | 3.30 | 0.85 |
| **Motor 1.1.0 (campeón actual)** | 4.68 | **3.19** | 0.83 |
| Motor con titulares CONFIRMED (1.0.0) | 4.83 | 3.09 | 0.87 |

**Error encontrado y corregido (champion/challenger):** el prior estructural de redistribución daba la mayor parte de los minutos del ausente a quien tenía más "espacio", o sea, al final de la banca, y contaba como ausentes a jugadores ya traspasados. Los jugadores con 83 % de DNP recibían 4.4 min de media cuando su media real era 1.2. Esto le quitaba ~1.3 min a la rotación.

| Comparación pareada (bootstrap por partido, IC95) | ΔMAE | ΔCRPS | Decisión |
|---|---|---|---|
| Challenger A (prior por rol + ausentes recientes) vs 1.0.0 | −0.26 [−0.28, −0.24] | −0.04 [−0.05, −0.02] | Mejor que 1.0.0; superado por B |
| **Challenger B (A + P(DNP) agrupado) vs 1.0.0** | −0.44 [−0.47, −0.42] | −0.11 [−0.13, −0.09] | **PROMOVIDO → 1.1.0** |
| 1.1.0 vs baseline fuerte | +0.04 [−0.00, +0.09] (empate) | −0.22 [−0.25, −0.19] | Mejor distribución; media puntual empatada |

Errores que siguen abiertos (próximos challengers):
- Titulares CORE sobreestimados (+1.7 min) y su intervalo es estrecho (cobertura 0.75).
- Los titulares confirmados mejoran el MAE en 0.30 min: la ventana de 30 minutos antes del juego tiene valor real.

## 4. Posesiones → oportunidades → eficiencia → Monte Carlo conjunto → props

Componentes nuevos:
- `sim/rates.py`, `NBA_OPPORTUNITY_RATES_V1`: tasas por posesión en cancha, con shrinkage bayesiano y pesos de recencia. También pace del equipo, usos por posesión, rebote, % de canastas asistidas, ratios defensivos del rival (perfil de tiro, % por zona, TOV forzadas, faltas) y ventaja de local medida (no supuesta).
- `sim/game_sim.py`, `NBA_FULL_PLAYER_MONTE_CARLO_V1`: guion latente → minutos → posesiones → usos → FGA/TOV/FT → tipo de tiro → conversión → TL → puntos → rebotes de los fallos que ocurrieron → asistencias de las canastas que ocurrieron → marcador.
- La contabilidad se cumple en el **100 % de las simulaciones** de 360 partidos reales: Σ pts = marcador, OREB + DREB rival ≤ fallos, AST ≤ canastas de compañeros, minutos = 240 + 25·OT.
- La simulación pura no tiene "score effects", y su margen salía con SD de 20.9 frente a 14.2 real. Se calibra re-muestreando simulaciones completas, que siguen siendo coherentes, para que el margen y el total tengan la varianza medida en partidos reales. Después de calibrar, la correlación local/visitante sale +0.21; en datos reales es +0.22, y **no se ajustó directamente**.
- Rendimiento: ~0.8 s por partido con 10,000 sims en 1 CPU. Cero simulaciones por request público.

Validación walk-forward, 47 fechas, **360 partidos, 7,639 jugador-mercado por mercado**, con minutos 1.1.0 y guion anclado al rating. Las líneas son pseudo-líneas (piso del promedio de los últimos 10 + 0.5), porque no hay líneas históricas gratuitas:

| Mercado | MAE modelo / baseline | CRPS modelo / baseline | Cobertura 80 % | Brier P(over) modelo / baseline | Brier calibrado (fuera de muestra, feb-abr) |
|---|---|---|---|---|---|
| PTS | **4.58** / 4.70 | **3.19** / 3.47 | 0.82 | 0.238 / 0.255 | 0.231 → **0.230** (baseline 0.257) |
| REB | **1.91** / 1.93 | **1.32** / 1.42 | 0.85 | 0.242 / 0.252 | 0.242 → **0.235** (0.255) |
| AST | **1.38** / 1.39 | **0.94** / 1.00 | 0.89 | 0.234 / 0.248 | 0.231 → **0.228** (0.249) |
| 3PM | 0.92 / **0.90** | **0.59** / 0.63 | 0.93 | 0.214 / 0.217 | 0.214 → **0.212** (0.219) |
| PRA | **6.02** / 6.21 | **4.23** / 4.62 | 0.78 | 0.239 / 0.255 | 0.233 → **0.228** (0.257) |

Con la promoción de minutos 1.1.0, los props mejoraron en todos los mercados respecto a la corrida con 1.0.0 (PTS: MAE 4.62 → 4.58, CRPS 3.22 → 3.19).

Lectura honesta:
- El modelo gana en distribución (CRPS) en los 5 mercados y en media puntual en 4 de 5. En 3PM la media queda ligeramente peor.
- Es **sobreconfiado en los extremos**: cuando dice 84 %, ocurre 68 %. `NBA_PROB_CALIBRATION_V1` (logística por mercado, ajustada solo con pronósticos ya liquidados) mejoró el Brier fuera de muestra en los 5 mercados. Queda en SHADOW: se publica junto al valor crudo, sin reemplazarlo.
- El error de puntos lo dominan los minutos: cuando los minutos se sobreestiman >8, el sesgo en puntos es +6.6. Esto confirma la arquitectura MINUTES-FIRST.

## 5. Game markets

| Ancla del guion | MAE margen | CRPS margen | Brier victoria local | MAE total |
|---|---|---|---|---|
| Bottom-up (jugadores) | 11.51 | 8.26 | 0.215 | 15.00 |
| Baseline rating de diferencial de puntos | 11.12 | — | 0.210 | 15.04 |
| **Simulación anclada al rating (campeón; minutos 1.1.0)** | 11.13 | **8.08** | 0.209 | **14.98** |

Hoy los mercados de partido valen lo mismo que un rating simple de diferencial de puntos: **no hay ventaja demostrada**. La simulación conjunta sí aporta coherencia: props, minutos y marcador salen del mismo estado de juego. Siguiente challenger: rating ajustado por rival y por disponibilidad, usando on/off de jugadores reconstruido de los stints.

## 6. Correr la temporada 2026-27 sin gastar

- **GitHub Actions en un repo público:** minutos gratis sin tarjeta. Un cron cada 15-30 min en días de juego ejecuta `nba-platform tick`, para CDN y el PDF de lesiones. La base SQLite se guarda como artifact o release.
  - Riesgo: el CDN o los PDFs podrían rechazar IPs de Azure. Se verifica en la primera corrida. Si fallan, se usa la opción siguiente.
- **Tu Mac con `launchd` + SQLite local:** IP residencial, sin costo, y desde ahí `nba_api` / stats.nba.com sí suele responder. Requiere que la Mac esté encendida en las ventanas de juego.
- **Descartados por la regla $0:** Oracle Free, Fly.io, Render y otros piden tarjeta. Los proxies no se usan.
- **Licencia:** los datos de NBA.com son para uso personal, no comercial. Si la página se vuelve pública o comercial, hay que revisarlo.

## 7. Qué no se puede hacer gratis (arquitectura lista, en espera)

| Función | Limitación | Estado |
|---|---|---|
| Comparación con sportsbooks / EV / CLV | No hay líneas históricas gratuitas. The Odds API tiene un tier gratis (registro sin tarjeta, 500 créditos/mes), pero no es accesible desde aquí y solo sirve para capturas futuras. | `compare_to_book()` y la tabla `market_lines` listos; nunca se llaman con cuotas inventadas |
| Tracking real (touches, potential assists) | stats.nba.com bloquea la nube | Estimadores observables (§2) en producción |
| Inactivos oficiales históricos | No vienen en el PBP | Inferidos y etiquetados; en vivo, el PDF oficial |
| 2026-27 en este entorno | Egress bloqueado; el dataset abierto aún no lo incluye | Pipeline listo para el runner gratuito |
| Steals/blocks props | Prioridad baja según la especificación | Datos ya reconstruidos (`stl`, `blk`) |

## 8. Registro champion/challenger

Ver `models/registry.json`: hipótesis, configuración, métricas reales e IC de cada promoción y rechazo.
