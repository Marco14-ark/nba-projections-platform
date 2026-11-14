"""NBA_PROB_CALIBRATION_V1 — logistic recalibration of P(over) per market (spec §52).

Fitted ONLY on past, already-settled forecasts (walk-forward). Raw model probabilities are kept;
calibrated ones are published alongside, never silently replacing them. On real 2025-26 data the
fitted slopes were 0.38-0.80 (the raw simulation is over-confident at the extremes), and the
recalibration lowered out-of-sample Brier in all five markets tested.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

CAL_NAME = "NBA_PROB_CALIBRATION_V1"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_logistic(p: np.ndarray, y: np.ndarray, min_n: int = 500) -> tuple[float, float] | None:
    if len(p) < min_n:
        return None
    x, y = _logit(p), np.asarray(y, float)

    def nll(ab):
        q = 1 / (1 + np.exp(-(ab[0] + ab[1] * x)))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    a, b = minimize(nll, [0.0, 1.0]).x
    return float(a), float(b)


def apply(p, ab: tuple[float, float] | None):
    if ab is None:
        return np.asarray(p, float)
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * _logit(p))))


class MarketCalibrator:
    def __init__(self):
        self.params: dict[str, tuple[float, float] | None] = {}

    def fit(self, rows, before_date: str) -> "MarketCalibrator":
        """rows: DataFrame with market, date, p_over, over (settled forecasts)."""
        past = rows[(rows["date"] < before_date)].dropna(subset=["p_over", "over"])
        for mk, d in past.groupby("market"):
            self.params[mk] = fit_logistic(d["p_over"].to_numpy(), d["over"].to_numpy())
        return self

    def __call__(self, market: str, p):
        return apply(p, self.params.get(market))
