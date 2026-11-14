"""NBA_PROB_CALIBRATION_V2 — calibrators for P(over) trained ONLY on earlier, settled forecasts.

Methods (all per-row inputs are pregame quantities from the frozen forecast):
  RAW           model probability
  PLATT_GLOBAL  logistic on logit(p)
  PLATT_MKT     logistic on logit(p), one per market
  BETA_MKT      logistic on [log p, log(1-p)] per market (beta calibration)
  ISO_MKT       isotonic regression per market (calibration by probability range)
  HIER          one regularised logistic model: logit(p) x {market, minutes stability (CV), expected
                starter, P(DNP), absent rotation minutes (injury context), log sample size} +
                market intercepts. Groups with little data borrow strength through the shared
                coefficients (L2 shrinkage) — hierarchical in the empirical-Bayes sense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
MARKETS = ["PTS", "REB", "AST", "3PM", "PRA"]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def long_format(P: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, market, line): base pseudo-line + offset lines."""
    base_cols = ["date", "game_id", "player_id", "market", "minutes_pred", "minutes_sd", "p_dnp_pred", "exp_starter",
                 "absent_rot_min", "n_games", "flag_return", "flag_new_team"]
    out = [P[base_cols + ["p_over", "over"]].assign(offset=0)]
    for c in P.columns:
        if c.startswith("p_over_") and c[7:] not in ("",):
            o = c[7:]
            y = "over_" + o
            if y in P:
                out.append(P[base_cols + [c, y]].rename(columns={c: "p_over", y: "over"}).assign(offset=int(o)))
    L = pd.concat(out, ignore_index=True).dropna(subset=["p_over", "over"])
    L["cv"] = (L["minutes_sd"] / L["minutes_pred"].clip(lower=1)).clip(upper=3)
    L["logn"] = np.log1p(L["n_games"].clip(lower=0))
    L["absent"] = L["absent_rot_min"] / 48.0
    return L


def _hier_X(L: pd.DataFrame) -> np.ndarray:
    z = logit(L["p_over"])
    cols = [z]
    for m in MARKETS:
        d = (L["market"] == m).to_numpy(float)
        cols += [d, d * z]
    for f in ("cv", "exp_starter", "p_dnp_pred", "absent", "logn"):
        v = L[f].to_numpy(float)
        cols += [v, v * z]
    return np.column_stack(cols)


class Calibrator:
    def __init__(self, method: str):
        self.method = method
        self.models: dict = {}

    def fit(self, L: pd.DataFrame) -> "Calibrator":
        m = self.method
        if m == "RAW":
            return self
        if m == "PLATT_GLOBAL":
            self.models["*"] = LogisticRegression(C=1e4).fit(logit(L.p_over)[:, None], L.over)
        elif m == "HIER":
            self.models["*"] = LogisticRegression(C=1.0, max_iter=2000).fit(_hier_X(L), L.over)
        else:
            for mk, d in L.groupby("market"):
                if len(d) < 300:
                    continue
                if m == "PLATT_MKT":
                    self.models[mk] = LogisticRegression(C=1e4).fit(logit(d.p_over)[:, None], d.over)
                elif m == "BETA_MKT":
                    p = np.clip(d.p_over.to_numpy(), EPS, 1 - EPS)
                    self.models[mk] = LogisticRegression(C=1e4).fit(np.column_stack([np.log(p), -np.log(1 - p)]), d.over)
                elif m == "ISO_MKT":
                    self.models[mk] = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(d.p_over, d.over)
        return self

    def predict(self, L: pd.DataFrame) -> np.ndarray:
        m = self.method
        p = L["p_over"].to_numpy(float)
        if m == "RAW":
            return p
        if m == "PLATT_GLOBAL":
            return self.models["*"].predict_proba(logit(p)[:, None])[:, 1]
        if m == "HIER":
            return self.models["*"].predict_proba(_hier_X(L))[:, 1]
        out = p.copy()
        for mk in L["market"].unique():
            sel = (L["market"] == mk).to_numpy()
            mod = self.models.get(mk)
            if mod is None:
                continue
            if m == "PLATT_MKT":
                out[sel] = mod.predict_proba(logit(p[sel])[:, None])[:, 1]
            elif m == "BETA_MKT":
                q = np.clip(p[sel], EPS, 1 - EPS)
                out[sel] = mod.predict_proba(np.column_stack([np.log(q), -np.log(1 - q)]))[:, 1]
            elif m == "ISO_MKT":
                out[sel] = mod.predict(p[sel])
        return np.clip(out, EPS, 1 - EPS)


def ece(p, y, bins=10) -> float:
    p, y = np.asarray(p), np.asarray(y)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            tot += s.sum() * abs(p[s].mean() - y[s].mean())
    return float(tot / len(p))


def brier(p, y):
    return (np.asarray(p) - np.asarray(y)) ** 2


def logloss(p, y):
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward(L: pd.DataFrame, blocks: list[tuple[str, str]], methods: list[str], min_train: int = 5000) -> pd.DataFrame:
    """For each [start, end) block: fit on rows with date < start, predict the block."""
    out = []
    for a, b in blocks:
        tr, te = L[L["date"] < a], L[(L["date"] >= a) & (L["date"] < b)]
        if len(tr) < min_train or not len(te):
            continue
        res = te[["date", "game_id", "player_id", "market", "offset", "over", "p_over"]].copy()
        res["block"] = a
        for m in methods:
            res["p_" + m] = Calibrator(m).fit(tr).predict(te)
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------- production
import json as _json

PARAMS_PATH = "models/calibration_beta_mkt.json"


def fit_production(L: pd.DataFrame, fitted_through: str, path: str = PARAMS_PATH) -> dict:
    """Fit BETA_MKT on every settled forecast (dates < fitted_through) and save versioned params."""
    tr = L[L["date"] < fitted_through]
    out = {"name": "NBA_PROB_CALIBRATION_V2", "method": "BETA_MKT", "fitted_through": fitted_through,
           "n_rows": int(len(tr)), "markets": {}}
    for mk, d in tr.groupby("market"):
        p = np.clip(d.p_over.to_numpy(), EPS, 1 - EPS)
        lr = LogisticRegression(C=1e4).fit(np.column_stack([np.log(p), -np.log(1 - p)]), d.over)
        out["markets"][mk] = {"a": float(lr.coef_[0][0]), "b": float(lr.coef_[0][1]), "c": float(lr.intercept_[0]), "n": int(len(d))}
    out["version"] = f"BETA_MKT@{fitted_through}"
    with open(path, "w") as f:
        _json.dump(out, f, indent=1)
    return out


def apply_production(market: str, p, params: dict):
    m = params["markets"].get(market)
    if m is None:
        return np.asarray(p, float)
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = m["a"] * np.log(q) - m["b"] * np.log(1 - q) + m["c"]
    return 1 / (1 + np.exp(-z))
