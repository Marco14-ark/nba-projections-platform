# FASE V4: integridad de escenarios y validación prospectiva

Fecha: 2026-11-14. Presupuesto: $0 MXN. **Champion: STACK_V2_ENV (sin cambios).**
Tests: **75 passed** (18 nuevos en `tests/test_v4.py`).
Reproducibilidad: después de todos los cambios de V4 volví a correr el Champion sobre 2026-27 (121 partidos) y salió **bit a bit idéntico** (diferencia máxima 0.0 en props y en mercados de juego).

Las tres categorías de evaluación nunca se mezclan:

| Categoría | Dónde vive | Qué contiene |
|---|---|---|
| Backtest histórico | `data/m4_*.pkl`, `data/st_*.pkl`, `data/mc4_*.pkl` | Walk-forward, siempre sin fugas |
| Shadow (challengers) | `data/live/shadow_STACK_V3_SIGMA.db`, `data/live/shadow_STACK_V4_BLOWOUT.db` | Snapshots reales congelados antes del inicio, con `category=SHADOW` |
| Prospectivo (Champion) | `data/live/prospective.db` | Snapshots reales del Champion, con `category=PROSPECTIVE` |

---

## 1. Corrección de escenarios

**El problema.** La liquidación V1 tomaba el primer escenario (`H0A0` / `S0`, es decir, "todos los dudosos juegan"), aunque lo publicado no fuera eso.

**La solución.** Cada snapshot guarda, en la misma transacción, un registro inmutable en `forecast_publications` por componente: MINUTES (por equipo), PROPS y GAME. Hay tres tipos de publicación:

| Tipo | Cuándo | Qué se publica y qué se liquida |
|---|---|---|
| `SINGLE` | Un solo escenario de disponibilidad | Las filas `BASE` |
| `MIXTURE` | Todos los estados enumerados tienen una probabilidad empírica justificada y la ventana está a ≤240 min del inicio | La mezcla exacta a nivel de simulación (`MIXTURE` en minutos, `MIX` en props y juego). Los pesos se congelan con hash y fuente antes del partido. Se liquida esa mezcla con esos pesos, gane quien gane. |
| `CONDITIONAL` | No hay pesos justificados | Pronósticos condicionales separados ("si X juega" / "si X no juega"), sin mezcla. Se liquida **solo** el escenario cuyas condiciones ocurrieron según el box oficial (el jugador estaba activo o no). Si la disponibilidad no consta, el evento queda como `NO_MATCHING_SCENARIO` y no se adivina. |

Cómo lo hace el código:
- `settle_published()` lee únicamente la etiqueta publicada.
- Las tablas `settlements` y `settlement_events` son inmutables.
- Cada fila liquidada lleva `category`, `stack`, `settle_method` y `weights_hash`.
- `forecast_results` (V1) se conserva porque es inmutable, pero queda reemplazada.

**Mezcla exacta de props y mercados de juego.** `sim/mixture.py` asigna cada simulación a un escenario con una semilla fija y con los pesos ya congelados. Un jugador ausente de un escenario aporta 0 en esas simulaciones. El resultado es un `GameSim` normal: la contabilidad se mantiene y se resume con las mismas funciones de mercado.

**Probabilidades de los escenarios** (`models/play_rates_hist.json`):
- **Origen.** Son reportes oficiales de lesiones de la NBA, gratuitos. Bajé 321 PDFs, del 2025-12-22 al 2026-11-12. Los anteriores a diciembre de 2025 ya no están disponibles (403), así que no los usé. La descarga fue espaciada, una petición cada 1.5 s.
- **Resultado medido.** Si el jugador apareció o no en ese partido según el box oficial.
- **Resolución de nombres.** Contra el roster del equipo en toda la temporada, no contra el box de ese partido. Hacerlo contra el box sesgaba P(juega) hacia arriba: la primera versión daba OUT→jugó 8%, lo cual es imposible.
- **Ajuste.** Con partidos del 2025-12-22 al 2026-04-12.
- **Validación fuera de muestra.** En 2026-27:

| Estado | n (ajuste) | P(juega) ajustada | n (2026-27) | Observado 2026-27 |
|---|---|---|---|---|
| QUESTIONABLE | 733 | 0.509 | 175 | 0.480 |
| DOUBTFUL | 183 | 0.006 | 29 | 0.034 |
| PROBABLE | 323 | 0.951 | 54 | 0.870 |
| OUT | 4,562 | 0.002 | 787 | 0.000 |

Reglas de uso:
- Las tasas se midieron a ~100 min del inicio (mediana; p90 160 min). Por eso solo se aplican a ventanas a ≤240 min del inicio. INITIAL y 6H siguen siendo `CONDITIONAL`.
- En cuanto el archivo propio de reportes capturados tenga n≥40 por estado, tiene prioridad sobre las tasas históricas.

## 2. Auditoría de los snapshots existentes

Ejecuté `scripts/audit_snapshots.py` en las tres bases. Los registros de publicación de los snapshots anteriores a V4 se reconstruyeron solo a partir de lo almacenado y quedaron marcados `backfilled=1`.

| Base | Snapshots | Después del inicio | Con escenarios CONDITIONAL | MIXTURE | Problemas de coherencia |
|---|---|---|---|---|---|
| prospective.db | 28 | 0 | 7 | 0 | 0 |
| shadow_STACK_V3_SIGMA.db | 411 | 0 | 5 | 0 | 0 |
| shadow_STACK_V4_BLOWOUT.db | 22 | 0 | 0 | 0 | 0 |

- Ningún snapshot anterior a V4 publicó una mezcla: todas las probabilidades de escenario eran `None`. Por lo tanto, todos los casos QUESTIONABLE eran `CONDITIONAL`.
- La liquidación V1 era incorrecta en **OKC@LAC**. El jugador dudoso de OKC no jugó, pero V1 liquidó el escenario en que jugaba. V4 liquida `H0A1` / `S1`.
- En GSW@SAS los dos dudosos jugaron, así que `H0A0` coincidía por casualidad.
- Con la liquidación correcta, el MAE de minutos de los 3 partidos baja de **6.42 a 5.85** y la cobertura P10–P90 sube de 0.685 a 0.722.
- Coherencia verificada:
  - Ningún jugador OUT tiene minutos.
  - La suma de minutos por escenario está entre 240 y 250.
  - Todos los snapshots se crearon antes del inicio.
  - Archivo 2026-27: 183 partidos, 0 duplicados, 0 líneas corruptas y termina en salto de línea.
  - Todos los box finales cuadran: minutos = 240+25·OT y la suma de puntos = marcador.

## 3. Diagnóstico de blowouts

**Medición.** El tiempo basura (G) se mide desde el PBP, con una regla fijada de antemano y no ajustada a resultados: minutos del 4.º cuarto con ventaja ≥25 (quedando 12–9 min), ≥20 (9–6 min) o ≥15 (<6 min). Así se midieron 2,643 partidos de 2024-25, 2025-26 y 2026-27.

| \|margen final\| | Partidos | G medio | % con G>0 |
|---|---|---|---|
| 0–5 | 644 | 0.1 | 4% |
| 10–15 | 488 | 2.9 | 64% |
| 15–20 | 333 | 5.9 | 94% |
| 20–25 | 214 | 8.1 | 100% |
| 30–40 | 131 | 10.7 | 100% |

**Patrón histórico del Champion** (2025-26). Sesgo = predicción − real, según el G que realmente ocurrió (dato usado solo para diagnóstico, nunca como input):

| G real | Sesgo titulares | Sesgo marginales |
|---|---|---|
| 0 | −1.67 min | +0.96 |
| 9–12 | **+3.01 min** | **−2.44** |

Los 3 primeros partidos siguen exactamente ese patrón:

| Partido | G | Sesgo titulares | Sesgo suplentes |
|---|---|---|---|
| GSW@SAS (+38) | 12 | +6.1 | −3.1 |
| POR@SAC | 0 | −3.3 | +2.7 |
| OKC@LAC | 0 | −2.6 | +2.0 |

**Conclusión.** La mayor parte del error es irreducible antes del partido: no se sabe si habrá paliza. Lo que sí se puede mejorar es la *forma* de la mezcla. El MAE de 6.4 venía de eso, más el error de liquidación de OKC@LAC (punto 2).

**Challenger `BLOWOUT_GD`** (`MinutesConfig(blowout_feature="garbage", dnp_garbage=True)`):
- **Probabilidad pregame de paliza y trayectorias de ventaja.** El margen simulado lleva a G. G se toma de la distribución empírica de G dado |margen|, usando solo partidos de entrenamiento. No se usa el margen final real.
- **Cuándo sacan los entrenadores a los titulares.** Pendiente por equipo sobre G, contraída hacia la liga. En la liga, cada titular pierde 0.46 min por minuto de garbage. Los que más retiran: SAS −0.69, DET −0.67, DEN −0.65. Los que menos: UTA −0.16, MEM −0.09.
- **Rotaciones de garbage y minutos extra de suplentes.** P(DNP) de los marginales depende de G: logit −0.20 por minuto. Con 12 min de garbage, el logit baja 2.4.

**Validación** (walk-forward, mismos jugador-partido, IC 95% por bootstrap de partidos):

| Cohorte fuera de muestra | Filas | ΔMAE | ΔCRPS |
|---|---|---|---|
| 2024-25 (263 partidos) | 7,508 | **−0.031 [−0.043, −0.020]** | **−0.011 [−0.019, −0.003]** |
| 2025-26 (255 partidos) | 7,126 | **−0.019 [−0.032, −0.007]** | **−0.016 [−0.024, −0.008]** |
| 2025-26, otras fechas (256) | 7,089 | −0.007 [−0.019, +0.006] | −0.003 [−0.012, +0.005] |
| 2026-27 (121 partidos) | 3,192 | −0.015 [−0.041, +0.011] | −0.004 [−0.022, +0.013] |

Componentes por separado:
- Solo G: significativo solo en 2025-26.
- Solo DNP-garbage: significativo solo en 2024-25.

Efecto en props (stack completo frente a STACK_V2_ENV):
- 2025-26 (363 partidos): CRPS de PTS −0.005 [−0.010, −0.001]; resto no significativo.
- 2026-27 (121 partidos): Brier de REB +0.0010 [+0.0002, +0.0018], es decir, peor; resto no significativo.

**Decisión: shadow, no se promueve.** Las mejoras de minutos van todas en la misma dirección, pero solo son significativas en 2 de 4 cohortes, y en props el efecto es mixto. Ya corre prospectivamente en `shadow_STACK_V4_BLOWOUT.db`.

## 4. Resultados de los challengers Monte Carlo

Los parámetros se estimaron solo con 2024-25 (`scripts/estimate_joint_v2.py`). No apunté a ninguna correlación objetivo.
- **A:** factor del jugador σ=0.127, estimado por momentos cruzados de residuos por posesión.
- **B:** shock de usos del equipo σ=0.040 + asistencias ligadas a los tiros anotados por compañeros.
- **C:** A y B combinados.

Las evaluaciones usan los mismos partidos y la misma semilla: 2025-26 (102 partidos) y 2026-27 (56 partidos).

**Correlaciones:**

| Correlación | Real 25-26 | Champion | A | B | C |
|---|---|---|---|---|---|
| PTS–REB | 0.258 | 0.134 | 0.194 | 0.135 | 0.196 |
| PTS–AST | 0.165 | 0.152 | 0.201 | **0.049** | 0.093 |
| REB–AST | 0.200 | 0.097 | 0.142 | 0.086 | 0.124 |
| Compañeros PTS | −0.003 | −0.065 | −0.074 | −0.066 | −0.074 |

**Precisión individual** (Δ frente al Champion; las celdas en negrita son significativas):

| Reto | 2025-26 | 2026-27 |
|---|---|---|
| A | **Brier REB +0.0024 (peor)** | CRPS AST +0.003, **Brier PRA +0.0015 (peor)** |
| B | **Brier AST −0.002 (mejor)** | **CRPS PRA +0.012 (peor)** |
| C | **Brier PTS +0.001 (peor)** | **CRPS PRA +0.012 (peor)** |

Decisiones:
- **A:** mejora la estructura en PTS–REB y REB–AST, pero se pasa en PTS–AST y empeora algunas métricas individuales → no se promueve; queda como investigación.
- **B:** lleva PTS–AST en la dirección equivocada → rechazado.
- **C:** mixto → no se promueve.
- **Diagnóstico extra:** con los minutos del challenger de blowouts, la correlación entre compañeros pasa de −0.058 a −0.048 (real −0.027), pero las props no mejoran.

Hallazgo de fondo: los momentos cruzados no son consistentes con un factor único de cargas iguales. La covarianza entre uso y asistencias es casi nula, mientras que entre rebotes y los otros dos es mayor que el exceso de varianza de rebotes.

## 5. Validación histórica

- El Champion es reproducible bit a bit.
- Todos los challengers se compararon en filas idénticas, con IC por bootstrap de partidos, en tres temporadas: 2024-25, 2025-26 y 2026-27, más fechas reservadas.
- La única mejora consistente del ciclo es la de minutos `BLOWOUT_GD`, y aun así no alcanza el umbral para promoverla.

## 6. Validación prospectiva

Snapshots liquidados: 3 partidos (13-nov), con V4 y la distribución publicada.

| Mercado | n | MAE Champion | Cob. 80% Champion | MAE shadow V3_SIGMA | Cob. 80% shadow V3_SIGMA |
|---|---|---|---|---|---|
| MIN | 72 | 5.85 | 0.72 | 5.86 | 0.72 |
| PTS | 64 | 5.29 | 0.84 | 5.28 | 0.84 |
| REB | 64 | 1.63 | 0.88 | 1.64 | 0.89 |
| AST | 64 | 1.42 | 0.84 | 1.42 | 0.84 |
| PRA | 64 | 6.67 | 0.86 | 6.66 | 0.86 |

- Ganador acertado en 2 de 3. El total real cayó dentro de P10–P90 en los 3 partidos y el margen real en 2 de 3.
- **La muestra es demasiado pequeña para cualquier conclusión.**
- Ciclo en marcha: `scripts/prospective_cycle.py` hace todo a partir de una captura del navegador integrado (`scripts/mac/capture_bundle.js`):
  1. Agrega los finales al archivo de forma segura.
  2. Reconstruye frames, segmentos y garbage.
  3. Liquida cada base por separado.
  4. Congela los snapshots del Champion y de 2 shadows sobre los mismos partidos.
- Ya funcionó hoy a las 07:43Z: congeló 8 INITIAL nuevos para el Champion y 22 para el nuevo shadow de blowouts.
- Verificaciones programadas (en esta sesión) a las 18:10Z (6H), 23:35Z (90M/30M) y 2026-11-15 07:05Z (liquidar el 14-nov).
- Las verificaciones requieren que la app de escritorio esté abierta. Si no lo está, se reporta y no se inventa nada.

## 7. Nuevos tests (`tests/test_v4.py`, 18)

- **Archivos y datos:**
  - Archivo sin salto de línea final.
  - Registros duplicados.
  - Coherencia minutos/marcador (240+25·OT, suma de puntos).
- **Escenarios y probabilidades:**
  - QUESTIONABLE sin tasas → CONDITIONAL, sin mezcla.
  - QUESTIONABLE con tasas → MIXTURE con pesos que suman 1, hash, fuente y media del jugador dudoso menor que en "juega".
  - Rechazo de pesos que no suman 1.
  - Las publicaciones son inmutables (no se pueden actualizar ni borrar).
  - Parseo y emparejamiento de condiciones; con disponibilidad desconocida no se adivina.
- **Liquidación:**
  - CONDITIONAL liquida solo el escenario que ocurrió (ambos casos), es idempotente y las liquidaciones son inmutables.
  - Una MIXTURE se liquida como se publicó, con su hash, aunque el resultado favorezca a un escenario.
  - Disponibilidad desconocida → `NO_MATCHING_SCENARIO`.
  - Un snapshot posterior al inicio nunca se liquida.
  - Las categorías PROSPECTIVE, SHADOW y REPLAY se mantienen separadas.
- **Jugadores OUT y titulares:** un OUT no aparece en ningún lado y se respetan los titulares confirmados.
- **Simulación:**
  - Mezcla exacta de simulaciones: contabilidad correcta, proporción de pesos y ceros para el jugador ausente.
  - Mixture de props y mercados de juego dentro del autopilot con simulación.
  - Con los switches de Joint MC apagados se reproduce el Champion exacto.
  - Medición de garbage y muestreador (la prórroga nunca es garbage).

## 8. Errores pendientes

1. **Reportes de lesiones anteriores al 22-dic-2025.** No están disponibles (403). Las tasas de estados raros, como DOUBTFUL con n=29 en validación, siguen siendo imprecisas.
2. **15% de los nombres del reporte sin resolver.** Son jugadores two-way o G-League sin minutos. Se excluyen y no se inventan.
3. **Tasas de QUESTIONABLE por horizonte.** Solo valen cerca del inicio; INITIAL y 6H siguen siendo condicionales. Falta un modelo de P(juega | estado, horas antes del inicio) usando varios reportes del mismo día.
4. **Correlación entre compañeros.** Sigue demasiado negativa en todas las variantes (−0.06 frente a ≈0).
5. **Automatización en la Mac.** Las verificaciones dependen de que la app de escritorio esté abierta. launchd/curl nativo sigue sin verificarse (igual que en V3).
6. **Shadow STACK_V3_SIGMA.** Su base contiene snapshots INITIAL de partidos lejanos (inmutables); en las comparaciones se usan solo los partidos comunes.
7. **El shadow de blowouts empezó hoy.** Todavía no tiene partidos liquidados.

## 9. Próxima mejora de mayor impacto

**Modelo de disponibilidad por horizonte temporal: P(juega | estado oficial, minutos antes del inicio, historial del jugador).**
- Se construye con varios reportes del mismo día: 15 y 30 min, 90 min y 4–6 h antes del inicio.
- Hoy toda ventana más temprana que 4 h publica escenarios condicionales separados, y los minutos de los dudosos son la mayor fuente de error evitable, por encima de los blowouts.
- Con tasas calibradas por horizonte, todas las ventanas (INITIAL, 6H, 90M, 30M) podrían publicar mezclas justificadas y liquidables.
- Se valida igual que lo demás: Brier y calibración fuera de muestra en 2026-27, y después en shadow prospectivo.
