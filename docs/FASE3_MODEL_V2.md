# NBA Platform — Model Improvement V2 (13-nov-2026)

Presupuesto: **$0**. Todos los challengers se compararon contra el champion en **los mismos partidos, fechas y semillas**, con IC95 por bootstrap agrupado por partido. Los resultados completos están en `models/registry.json`.

## 0. Datos nuevos y auditoría

**Temporada 2026-27 real (174 partidos, 20-oct → 12-nov).**
- Se capturó desde la red de tu Mac con el navegador integrado, sin costo, y se trajo al entorno de trabajo.
- Validación contra los **boxscores oficiales**: titulares 100 % iguales, puntos 100 % iguales, minutos dentro de 0.1 min en el 99.7 % de los jugador-partido.
- Disponibilidad **oficial** (lista de inactivos del boxscore), no inferida.

**Los 23 partidos con errores de contabilidad: se recuperaron 22.**
- Causa: un jugador **expulsado desde la banca** o con técnica generaba un evento, y el algoritmo lo contaba como un sexto jugador en cancha.
- Corregido (`NON_PRESENCE_ACTIONS`). Queda 1 partido con un hueco de 4.3 min en la fuente; sigue excluido.
- Los segmentos quinteto-vs-quinteto suman exactamente el marcador final en 2,634 de 2,634 partidos.

**Fuga de información encontrada y eliminada.**
- La inferencia de ausencias usaba la *última aparición del jugador en la temporada* para decidir si seguía en el roster, es decir, información futura.
- Nueva versión `infer_absences_asof`: el roster se conoce solo por partidos anteriores.
- Un test verifica que el resultado hasta una fecha no cambia si se agregan partidos posteriores.
- Todo V2 se evaluó sobre este dataset sin fuga (v2). En 2026-27 se usa la lista oficial (v3).

## 1. Minutes Engine V2

**El "+1.7 min en titulares principales" era un artefacto estadístico.** Se agrupaba por el pronóstico del propio modelo (tier = mediana predicha ≥ 28), lo que selecciona los casos sobreestimados (regresión a la media). Agrupando por información previa al partido (promedio de los últimos 10), el sesgo de los jugadores de muchos minutos es +0.2 a +0.3. **No se aplicó ningún ajuste.**

Patrones reales encontrados:

| Causa | Sesgo | ¿Se puede corregir antes del juego? |
|---|---|---|
| Titulares en partidos cerrados / blowouts de 30+ | −1.9 / +3.3 | No en la media (el margen final se desconoce); sí en la distribución vía guion simulado |
| Titulares en prórroga | −5.8 | No (la prórroga ocurre en ~4 % de los partidos) |
| Banca profunda vs rotación media (competencia por minutos) | +0.6 / −0.5 | **Sí → M_DNP** |
| Estacional (noviembre / abril) | −1.0 / +0.9 | Pendiente (fin de temporada, tanking) |

Challengers evaluados en 77 fechas de 2025-26 (15,873 jugador-partido, titulares predichos):

| Challenger | ΔMAE [IC95] | ΔCRPS [IC95] | Decisión |
|---|---|---|---|
| **M_DNP**: competencia por minutos (logística: tasa propia de DNP + cuántos compañeros disponibles están arriba en la rotación) | **−0.248 [−0.264, −0.231]** | **−0.101 [−0.111, −0.091]** | **Promovido → 1.2.0** |
| M_OT: prórroga ∝ (participación)^5, ajustado con stints de 2024-25 | +0.003 [−0.000, +0.006] | 0.000 | Rechazado (sin efecto) |
| M_CLOSE: efecto de partidos cerrados | +0.045 [+0.036, +0.054] | +0.024 | Rechazado (peor) |
| M_SIGMA: σ corregida por grados de libertad del ridge | +0.007 | +0.002 | Rechazado. Lleva la cobertura de titulares a 0.786 pero sobrecubre el total |
| M_HL8 / M_HL18 (memoria de rotación) | +0.008 (n.s.) / +0.049 | −0.006 (n.s.) / +0.044 | Rechazados |

**Confirmación de M_DNP en las otras 76 fechas, nunca usadas para decidir:** ΔMAE −0.239 [−0.253, −0.225], ΔCRPS −0.092 [−0.100, −0.082]. La cobertura P10-P90 pasa de 0.828 a 0.806 (nominal 0.80).

Minutos 1.2.0 frente al baseline fuerte (últimos 10 disponibles): **MAE 4.32 vs 4.52, CRPS 2.97 vs 3.32**. Es la primera versión que le gana también en media puntual.

## 2. Injury-aware team rating (NBA_TEAM_RATING_V2)

**Componentes:**
- Impacto ofensivo y defensivo por jugador, por 100 posesiones, con RAPM sobre 84,532 segmentos quinteto-vs-quinteto reconstruidos del play-by-play. El control por la fuerza del rival está incluido en la regresión.
- Rating del equipo = suma, sobre los jugadores **disponibles**, de (minutos esperados / 48) × su impacto.
- Localía y descanso medidos en los datos: un back-to-back cuesta −2.3 a −2.8 puntos de margen.

**Proceso:**
- Regularización (1500) y pesos de mezcla ajustados **solo con 2024-25**.
- Prueba en **1,150 partidos de 2025-26**:

| Rating | MAE margen | Brier victoria |
|---|---|---|
| V1 diferencial de puntos (champion) | 11.483 | 0.2076 |
| V2 RAPM solo | 11.383 (Δ −0.10 [−0.30, +0.10], n.s.) | 0.2046 |
| **V2 mezcla 0.295·V1 + 0.845·RAPM** | **11.292 (Δ −0.19 [−0.36, −0.02])** | **0.2024 (Δ −0.0052 [−0.0091, −0.0014])** |

El total del RAPM tiene sesgo de −5.3, así que no se usa para totales.

## 3. Opportunity Engine V2

Se evaluó **con minutos y posesiones reales** (oráculo), para aislar el error de oportunidad del de minutos y del de eficiencia. Tuning en 2024-25 y prueba en 2025-26 (24,998 jugador-partido):

- **¿Cómo se redistribuyen las oportunidades cuando cambia la rotación?** Aproximadamente **en proporción al usage** de cada jugador en cancha (lo que ya hacía V1). Concentrarlas más en los jugadores de mucho usage (α = 1.5) es peor.
- **Memoria más larga de tasas (HL 45 partidos):**
  - Mejora 3PA (−1.2 %), FTA, OREB, DREB y AST, con IC < 0.
  - FGA y TOV quedan iguales.
  - Efecto pequeño pero consistente → integrado en la pila.

## 4. Monte Carlo V2 (diagnóstico estructural, 93 partidos reales)

| Propiedad | Simulación | Real |
|---|---|---|
| P(prórroga) | 3.8 % | 4.4 % |
| P(margen ≥ 20) | 22.4 % | 22.8 % |
| Dispersión de FGA de titulares | 3.7 | 3.6 |
| Dispersión de minutos de titulares | 4.7 | 5.4 (simulación demasiado estrecha) |
| Correlación de puntos entre titulares compañeros | −0.06 | +0.02 (la medida real incluye variación entre partidos) |

- Contabilidad correcta en el **100 %** de las simulaciones.
- **Se mantienen 10,000 simulaciones:** no hay evidencia de que más simulaciones mejoren la calidad. El error está en los modelos, no en el ruido Monte Carlo.

## 5. Validación de la pila completa (mismos partidos y semillas)

**Nuevo champion `STACK_V2_ENV`** = minutos 1.2.0 + tasas 1.1.0 (HL45 + entorno de liga) + guion anclado al rating V2.

**Entorno de liga.** En 2026-27 el anotado cayó a 226.8 pts por partido (frente a 231.2 en 2025-26), y todos los modelos arrastraban el nivel anterior (+4.6 de sesgo en totales). El ajuste compara pace y eficiencia de la temporada actual contra el entrenamiento, con shrinkage por partidos jugados.

| | 2025-26 (363 partidos, ~7,837 filas por mercado) | 2026-27 forward (121 partidos, ~2,741 filas) |
|---|---|---|
| PTS MAE / CRPS | 4.553→4.533 [−0.029, −0.011] / 3.167→3.163 (n.s.) | 4.537→4.474 [−0.082, −0.044] / 3.133→3.117 [−0.028, −0.003] |
| REB MAE / CRPS | 1.910→1.887 / 1.320→1.308 (ambos IC < 0) | 1.877→1.848 / 1.288→1.273 (ambos IC < 0) |
| AST MAE / CRPS | 1.377→1.365 / 0.932→0.927 (IC < 0) | 1.343→1.323 / 0.903→0.897 (IC < 0) |
| 3PM MAE / CRPS | 0.911→0.902 / 0.587→0.583 (IC < 0) | 0.918→0.905 / 0.588→0.584 (IC < 0) |
| PRA MAE / CRPS | 5.998→5.960 / 4.210→4.196 (IC < 0) | 5.946→5.861 / 4.146→4.114 (IC < 0) |
| Minutos MAE (solo jugadores que jugaron) | 4.904→4.875 [−0.043, −0.016] | 5.061→5.044 (n.s.) |
| **Margen MAE** | **11.125→10.820 [−0.61, −0.01]** | 11.813→11.593 (n.s.) |
| Total MAE | 15.031→15.085 (n.s.) | 15.217→15.081 (n.s.) |
| Brier victoria | 0.2097→0.2023 (IC [−0.015, +0.001]) | 0.1959→0.1941 (n.s.) |
| ECE P(over) | baja en 4 de 5 mercados, igual en PTS (p. ej. REB 0.065→0.054) | baja en 4 de 5, igual en PTS (REB 0.074→0.059) |

**Decisiones:**
- `STACK_V2` (sin entorno de liga) se rechazó: empeoró los totales en 2026-27 (+0.19 [+0.03, +0.34]).
- `STACK_V2_ENV` mejora o empata en todo en ambas temporadas, sin ninguna degradación significativa.
- Las mejoras en props son pequeñas (0.5–1.9 %) pero significativas y consistentes en dos temporadas.
- La calibración logística (`NBA_PROB_CALIBRATION_V1`) sigue en SHADOW: el ECE residual es de 3–6 puntos.

## 6. Automatización $0 — conectividad comprobada (no supuesta)

| Ruta | Prueba | Resultado |
|---|---|---|
| **Tu Mac, navegador integrado, con headers de nba.com** | fetch real | CDN schedule / boxscore / play-by-play **200**, stats.nba.com **200**, PDF oficial de lesiones del 13-nov 5:00 PM **200**. Navegación directa sin headers: 403 |
| VM Linux en tu Mac | curl | Sin red (allowlist) |
| Entorno de desarrollo en la nube | curl | Bloqueado (solo GitHub vía git) |
| GitHub Actions | no ejecutable aquí (sin token de GitHub válido) | **NO VERIFICADO**. Workflow `connectivity.yml` listo para correrlo primero |
| curl/python nativo en macOS | no ejecutable desde aquí | **NO VERIFICADO**. `scripts/daily_mac.sh` ejecuta `connectivity()` antes de todo y se detiene si algo falla |

Componentes listos:
- `pipeline/capture.py`: chequeo de conectividad y captura incremental de partidos finales al mismo formato JSONL.
- `scripts/daily_mac.sh`: captura, reconstrucción y snapshots.
- `com.nba-platform.daily.plist`: tarea `launchd` cada 30 min.

La ruta que **sí está verificada** es la captura por navegador, la que trajo 2026-27.

## 7. Pendientes (siguiente iteración)

- Starters-only σ (M_SIGMA solo para titulares) para cerrar la cobertura de 0.75.
- Efecto de fin de temporada / tanking (abril +0.9 min).
- Probabilidad de jugar para QUESTIONABLE con el archivo de PDFs oficiales que se empiece a capturar.
- Correlación entre compañeros (factor común de "noche del equipo").
- Promover la calibración cuando acumule cohortes de 2026-27.
