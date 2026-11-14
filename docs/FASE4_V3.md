# FASE V3 — Calibración de probabilidades + pronóstico en vivo

Fecha: 2026-11-13 (ET). Presupuesto: $0 MXN. Campeón: **STACK_V2_ENV** (se mantiene).
Tests: **57 passed** (`python -m pytest -q`), de ellos 10 nuevos en `tests/test_v3.py`.

---

## 1. Auditoría de reproducibilidad

| Verificación | Resultado |
|---|---|
| Re-ejecución completa del campeón STACK_V2_ENV, 2025-26 (walk-forward) | Idéntica bit a bit (mismas semillas, mismo hash de insumos) |
| Re-ejecución en 2026-27 (121 partidos, 27-oct → 12-nov) | Idéntica bit a bit (`st_STACK_V2_ENV_REPRO_*` vs `st_STACK_V2_ENV_*`) |
| `build_frames()` (constructor nuevo del archivo 2026-27) vs build v3 anterior | Igual, salvo una etiqueta (`"nan"` vs `"COACH"`) en un campo que no se usa |
| Fugas temporales | La disponibilidad 2026-27 viene del boxscore oficial. Las ausencias históricas usan `infer_absences_asof`, que solo mira juegos previos. La calibración se entrena siempre con fechas estrictamente anteriores. `fit_production` no usa la fecha de corte ni posteriores (hay test). |
| Snapshots | Las tablas tienen triggers que impiden UPDATE/DELETE. Cada snapshot guarda `input_hash` y un árbol de hashes. Settlement solo lee el último snapshot con `created_at < tip`. |

## 2. Resultados de calibración (NBA_PROB_CALIBRATION_V2)

- Datos: walk-forward mensual (dic-2025 → nov-2026), entrenamiento solo con meses previos.
- Muestra: 201,069 probabilidades over/under en 443 partidos; mercados PTS, REB, AST, 3PM y PRA con 2–4 líneas por jugador.

| Método | Brier | LogLoss | ECE |
|---|---|---|---|
| RAW (campeón sin calibrar) | 0.1926 | 0.5688 | 0.0282 |
| PLATT_GLOBAL | 0.1916 | 0.5644 | 0.0051 |
| PLATT_MKT | 0.1915 | 0.5640 | 0.0049 |
| **BETA_MKT** | **0.1915** | **0.5640** | **0.0037** |
| ISO_MKT | 0.1918 | 0.5652 | 0.0047 |
| HIER (mercado × rango × estabilidad de minutos × rol × disponibilidad) | 0.1914 | 0.5636 | 0.0051 |

BETA_MKT frente a RAW (bootstrap pareado por partido, IC 95%):
- Brier: −0.00116 [−0.00141, −0.00091]
- LogLoss: −0.00484 [−0.00577, −0.00389]
- Las tres métricas mejoran a la vez, sin empeorar ninguna. El modelo crudo era sobreconfiado.

ECE por mercado, RAW → BETA_MKT:

| Mercado | ECE RAW | ECE BETA_MKT |
|---|---|---|
| PTS | 0.032 | 0.006 |
| REB | 0.031 | 0.005 |
| AST | 0.020 | 0.006 |
| 3PM | 0.013 | 0.010 |
| PRA | 0.039 | 0.008 |

Decisiones:
- **BETA_MKT** se promueve como capa publicada. Las probabilidades crudas se siguen guardando junto a las calibradas.
- **HIER** queda en shadow: mejora Brier y LogLoss solo marginalmente, su ECE es peor y tiene más parámetros.
- **ISO** se rechaza.
- **Platt** se rechaza porque BETA_MKT lo domina.

Los parámetros de producción están en `models/calibration_beta_mkt.json` (versión `BETA_MKT@2026-11-13`). El diagrama de fiabilidad está en `docs/reliability_v3.png`.

## 3. Resultados de minutos (Minutes V3)

Sin inflar la SD de forma artificial. Dos hipótesis, cada una evaluada en dos mitades de fechas independientes de 2025-26.

| Modelo | Fechas pares (n=15,873): MAE / CRPS / cob80 / cob80 titulares | Fechas impares (n=16,645): MAE / CRPS / cob80 / cob80 titulares |
|---|---|---|
| Campeón 1.2.0 | 4.321 / 2.966 / 0.812 / 0.755 | 4.365 / 2.996 / 0.806 / 0.745 |
| M3_EMP (ruido empírico) | 4.319 / 2.970 / 0.808 / 0.745 | — (rechazado) |
| **M3_SIGMA_ST** (corrección de grados de libertad a la SD, solo titulares) | 4.315 / 2.956 / 0.822 / 0.777 | 4.359 / 2.986 / 0.816 / 0.767 |

- **M3_EMP:** se rechaza. Empeora el CRPS y la cobertura de titulares.
- **M3_SIGMA_ST** en fechas impares (held-out), frente al campeón:
  - ΔMAE −0.0054 [−0.0082, −0.0026]
  - ΔCRPS −0.0109 [−0.0129, −0.0087]
  - La cobertura P10–P90 de titulares sube de 0.745 a 0.767.
  - Es una mejora real a nivel de minutos.
- **M3_SIGMA_ST** de punta a punta en props 2026-27 (121 partidos, 2,741 filas por mercado):
  - Minutos: MAE 5.044 → 5.041 (no significativo).
  - PTS: MAE +0.0055 [0.0005, 0.0106] (peor).
  - AST: MAE +0.0016 [0.0001, 0.0031] (peor).
  - PRA: Brier −0.0006 [−0.0011, −0.0001] (mejor).
  - Margen, total y ganador: no significativos.
- **Decisión: NO se promueve.** Mejora una parte y empeora otra, aunque las diferencias sean pequeñas. Corre en **shadow** sobre los mismos partidos prospectivos.

## 4. Resultados de oportunidades (Opportunity V3)

Cobertura P10–P90 y CRPS por mercado en 2026-27 (STACK_V2_ENV frente a un baseline de promedio reciente):

| Mercado | Cobertura 80% modelo | CRPS modelo | Cobertura baseline | CRPS baseline |
|---|---|---|---|---|
| PTS | 0.814 | 3.117 | 0.672 | 3.430 |
| PRA | 0.789 | 4.114 | 0.658 | 4.551 |
| REB | 0.870 | 1.273 | 0.723 | 1.388 |
| AST | 0.899 | 0.897 | 0.758 | 0.980 |
| 3PM | 0.927 | 0.584 | 0.811 | 0.623 |

Lectura:
- PTS y PRA tienen el ancho correcto.
- REB, AST y 3PM tienen cobertura por encima del 80% sobre todo por discreción: con conteos enteros, un intervalo inclusivo cubre más del 80%. No es dispersión excesiva que haya que corregir con una regla arbitraria.
- La traducción a probabilidades ya queda corregida por BETA_MKT, que ajusta precisamente esa sobreconfianza en las colas.
- Conclusión: no hay cambio estructural en las tasas de oportunidad en V3.

## 5. Diagnóstico Monte Carlo (10,000 simulaciones; no se aumentan)

93 partidos 2026-27, comparando lo real con lo simulado:

| Estadístico | Real | Simulado | Diagnóstico |
|---|---|---|---|
| P(prórroga) | 0.044 | 0.038 | OK |
| P(\|margen\| ≥ 20) (paliza) | 0.228 | 0.226 | OK |
| P(\|margen\| ≤ 5) (cerrado) | 0.245 | 0.267 | Ligeramente alto |
| corr(PTS, REB) mismo jugador | 0.262 | 0.131 | **Demasiado baja** |
| corr(PTS, AST) mismo jugador | 0.164 | 0.150 | OK |
| corr(REB, AST) mismo jugador | 0.201 | 0.096 | **Demasiado baja** |
| corr(MIN, PTS) | 0.525 | 0.374 | **Demasiado baja** |
| corr(PTS) entre titulares del mismo equipo | −0.004 | −0.070 | **Demasiado negativa** |

- El guion de partido (palizas, prórroga, margen) está bien calibrado.
- Falta un **factor común "forma del jugador en el partido"**: lo que hoy está ausente es un shock compartido de eficiencia y uso que suba o baje juntos sus PTS, REB y AST.
- El reparto multinomial de usos en el equipo obliga a que los compañeros compitan por los tiros, y eso exagera la correlación negativa entre ellos.
- Afecta a PRA y combinaciones del mismo partido, no a las props individuales.

## 6. Estado REAL de la automatización

| Componente | Estado |
|---|---|
| Captura desde el navegador integrado de la app de escritorio en la Mac (fetch con Origin nba.com) | **Verificado.** Calendario, boxscore, PBP, stats.nba.com y PDF de lesiones devolvieron 200. Bundle del 2026-11-13 capturado y validado. |
| Parser del PDF oficial real (diseño sin espacios) | **Verificado.** Reporte de 08:30 PM ET: 122 filas, 117 jugadores, 5 equipos "Not Yet Submitted". Hay test. |
| `live_forecast.py`: ventanas INITIAL/6H/90M/30M, recálculo por hash, rechazo después del inicio | **Verificado** en el sandbox con datos reales y tests. |
| `settle_finals.py` / `settle_game` (solo pregame, idempotente, inmutable) | **Verificado con tests**, todavía sin resultados reales (ningún partido de la cohorte ha terminado al momento de escribir esto). |
| `daily_run.py`: camino de "stop" cuando no hay conectividad | **Verificado.** |
| `curl`/Python nativos de macOS contra nba.com (`check_connectivity.sh`) | **NO verificado.** La terminal de la Mac a la que tuve acceso no tenía red. |
| launchd (`install_launchd.sh`) | **NO verificado** en tu Mac. |
| GitHub Actions | **NO probado.** El token de GitHub de la sesión era inválido. |

Instrucciones exactas para la Mac (Terminal, desde la carpeta del proyecto):

```bash
unzip nba-platform-v3.zip -d nba-platform && cd nba-platform
bash scripts/mac/bootstrap_data.sh ~/Downloads/nba-data-bootstrap.zip   # descomprime ./data y verifica SHA-256
python3 -m pip install --user numpy pandas scipy scikit-learn pdfplumber
python3 -m pytest -q                                                    # esperado: 57 passed
bash scripts/mac/check_connectivity.sh; echo "exit=$?"                  # esperado: todas las líneas 200 y exit=0
NBA_DATA_DIR=$PWD/data NBA_CODE_DIR=$PWD python3 scripts/mac/daily_run.py   # una corrida manual
bash scripts/mac/install_launchd.sh                                     # solo si los dos pasos anteriores funcionaron
launchctl list | grep nba-platform                                      # confirma que quedó cargado
```

Si `check_connectivity.sh` devuelve 403 en `stats.nba.com` desde curl, el respaldo gratuito ya probado es la captura desde el navegador integrado.

## 7. Primeros snapshots prospectivos (reales, antes del inicio)

Base: `data/live/prospective.db`. Contiene 20 snapshots de 17 partidos, creados el 2026-11-14 entre 01:36 y 01:42 UTC.

- Insumos de cada snapshot: reporte oficial de lesiones de 08:30 PM ET, rosters de los boxscores más recientes y titulares confirmados cuando ya existía el box pregame.
- Contenido guardado: 10,153 distribuciones de props.

| Partido (visitante@local) | Inicio UTC | Ventanas | Bandera |
|---|---|---|---|
| GSW@SAS | 11-14 02:30 | INITIAL → 90M | OK |
| POR@SAC | 11-14 03:00 | INITIAL → 90M | OK |
| OKC@LAC | 11-14 03:30 | INITIAL → 6H | OK |
| DEN@NOP, DAL@CLE, DET@MIA, BKN@MEM, CHI@MIN | 11-15 00:00–01:00 | INITIAL | PARTIAL_DATA |
| IND@PHI, LAL@PHX, CHA@LAC, SAS@UTA, DAL@BOS, NYK@TOR | 11-15 18:00–23:00 | INITIAL | PARTIAL_DATA |
| HOU@WAS, ATL@MIL, OKC@SAC | 11-16 00:00–02:00 | INITIAL | PARTIAL_DATA |

- PARTIAL_DATA significa que todavía no existía un reporte oficial para esa fecha. Es lo correcto: no se presenta como pronóstico definitivo.
- Ejemplo GSW@SAS (90M, escenario H0A0): margen local +8.5 (P10 −10, P90 +27), total 229.2 (P10 207, P90 252), P(gana SAS) 0.718. *(Corrección: la versión anterior de este documento atribuía a GSW@SAS las cifras de otro snapshot.)*
- Cada snapshot guarda: hora, versión del modelo y del calibrador, jugadores disponibles, lesiones conocidas, distribuciones de minutos, props, marcador y hash de insumos.
- `forecast_results`: liquidado el 2026-11-14 07:17 UTC (ver abajo). Nada se recalculó.

### 7b. Primera liquidación prospectiva (2026-11-14, 3 partidos)

Finales capturados con el navegador integrado (boxscore + PBP oficiales de los 9 partidos del 13-nov, archivo 2026-27 ahora con 183 partidos). Se liquidó solo el último snapshot anterior al inicio.

| Partido | Snapshot usado | Margen local pred. [P10, P90] | Real | Total pred. [P10, P90] | Real | P(gana local) | Ganó |
|---|---|---|---|---|---|---|---|
| GSW@SAS | 90M 01:42Z | +8.5 [−10, +27] | **+38** (fuera) | 229.2 [207, 252] | 208 | 0.718 | SAS ✔ |
| POR@SAC | 90M 01:42Z | −11.1 [−30, +8] | −13 | 232.0 [210, 255] | 227 | 0.230 | POR ✔ |
| OKC@LAC | 6H 01:42Z | −5.7 [−25, +13] | +1 | 222.4 [200, 245] | 225 | 0.349 | LAC ✘ |

Props (64 jugador-mercado por estadística, 73 de minutos):

| Mercado | MAE campeón | MAE shadow | Cob. 80% campeón | Cob. 80% shadow |
|---|---|---|---|---|
| MIN | 6.42 | 6.41 | 0.685 | 0.699 |
| PTS | 5.25 | 5.25 | 0.812 | 0.812 |
| REB | 1.68 | 1.68 | 0.859 | 0.859 |
| AST | 1.44 | 1.44 | 0.828 | 0.828 |
| 3PM | 0.95 | 0.95 | 0.891 | 0.891 |
| PRA | 6.79 | 6.79 | 0.844 | 0.844 |

Con 3 partidos no hay ninguna conclusión estadística. La cobertura de minutos baja (0.685) se explica en buena parte por la paliza de 38 puntos en SAS (titulares con pocos minutos). Es exactamente el tipo de error que la acumulación prospectiva debe medir. Nota: cuando el payload tiene varios escenarios de "Questionable", settlement liquida el primero (H0A0); hay que añadir el escenario mezclado ponderado (pendiente).

## 8. Champion vs Challengers

| Stack / componente | Rol | Evidencia |
|---|---|---|
| **STACK_V2_ENV** (minutos 1.2.0 + tasas 1.1.0 + rating V2 + BETA_MKT) | **CHAMPION** | Reproducible. Calibración V2 mejora Brier, LogLoss y ECE. |
| STACK_V3_SIGMA (minutos M3_SIGMA_ST) | SHADOW | Minutos mejor (significativo en ambas mitades). Props mixtas en 2026-27. Ya genera snapshots en `shadow_STACK_V3_SIGMA.db` sobre la **misma cohorte de 17 partidos**: medias de minutos ±0.12 y P10–P90 de titulares 12.8 → 13.4. |
| HIER (calibración) | SHADOW | Brier/LogLoss marginalmente mejores, ECE peor |
| M3_EMP, ISO_MKT, PLATT | RECHAZADOS | Ver secciones 2 y 3 |

La primera corrida shadow, sin límite de horizonte, también generó snapshots INITIAL para 394 partidos futuros. Eso ya se corrigió: `--horizon-days` vale 3 por defecto. La comparación prospectiva usa solo los partidos que ambos stacks tienen con snapshot antes del inicio.

Todo queda registrado en `models/registry.json`.

## 9. Errores pendientes

1. **Automatización nativa en la Mac sin verificar:** curl/launchd. La corrida diaria depende de que ejecutes `check_connectivity.sh` y el instalador.
2. **Settlement:** funcionó con resultados reales el 14-nov (3 partidos). Pendiente: liquidar contra la mezcla ponderada de escenarios en vez del primero (H0A0); y el archivo JSONL no terminaba en salto de línea, lo que rompió la primera concatenación (corregido: se reescribe el archivo completo).
3. **10 jugadores del reporte oficial sin resolver:** novatos o sin minutos NBA (Veesaar, Diop, Sorber, "Udeh, Jr., Ernest"…). Se guardan como no resueltos.
4. **Correlaciones dentro del jugador y entre compañeros** mal calibradas (sección 5).
5. **El archivo de probabilidades de "Questionable"** se basa en etiquetas históricas reconstruidas, no en reportes oficiales archivados.
6. **Snapshots de partidos lejanos en la base shadow:** no se pueden borrar porque son inmutables. Quedan marcados por la ventana y la fecha de creación, y se excluyen de la comparación.
7. **GitHub Actions** sin probar (token).

## 10. Siguiente mejora de mayor impacto

**Factor latente compartido "forma del jugador en el partido" en el Monte Carlo (MC V4):**
- Un shock lognormal por jugador y partido que mueva juntos uso, eficiencia y minutos.
- Desacoplar parcialmente el reparto multinomial de usos entre compañeros.
- Calibrar su varianza para reproducir las correlaciones reales: PTS-REB 0.26, REB-AST 0.20, MIN-PTS 0.53 y compañeros ≈0.
- Es lo único con sesgo estructural claro y medido.
- Afecta directamente a PRA y combinaciones del mismo partido.
- Se valida igual que lo demás: en shadow, con IC pareados, sobre 2025-26, 2026-27 y la cohorte prospectiva, una vez que tenga resultados.
