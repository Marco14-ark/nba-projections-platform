"""Odds arithmetic. American odds are integers (e.g. -110, +125)."""
from __future__ import annotations
import math


def american_to_decimal(a: float) -> float:
    a = float(a)
    if a == 0 or (-100 < a < 100):
        raise ValueError(f"invalid American odds {a}")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied_prob(a: float) -> float:
    """Raw implied probability (includes the vig)."""
    return 1 / american_to_decimal(a)


def fair_american(p: float) -> float | None:
    """Fair (no-vig) American price for probability p of winning (pushes excluded)."""
    if p is None or not (0 < p < 1):
        return None
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def no_vig_pair(a_side: float, a_other: float) -> tuple[float, float, float]:
    """Multiplicative (proportional) de-vig of a two-way market.
    Returns (p_side, p_other, overround)."""
    i, j = implied_prob(a_side), implied_prob(a_other)
    s = i + j
    return i / s, j / s, s - 1


def ev_per_unit(p_win: float, p_push: float, a: float) -> float:
    """Expected profit per 1 unit staked. A push refunds the stake (0 profit)."""
    p_lose = max(0.0, 1.0 - p_win - p_push)
    return p_win * (american_to_decimal(a) - 1) - p_lose


def fair_american_with_push(p_win: float, p_push: float) -> float | None:
    """Break-even American price when pushes refund: uses P(win | no push)."""
    q = 1 - p_push
    if q <= 0:
        return None
    return fair_american(p_win / q)


def mc_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))
