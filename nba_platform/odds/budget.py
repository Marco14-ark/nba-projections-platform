"""Free-plan budgeting. The monthly allowance (minus a safety reserve) is spread evenly over the
remaining days of the month; a fetch is allowed only if its estimated cost fits today's share."""
from __future__ import annotations
import calendar
from datetime import datetime


def allowance(limit: int, used_month: int, used_today: int, now: datetime, reserve: float = 0.10) -> dict:
    eff = int(limit * (1 - reserve))
    days_in = calendar.monthrange(now.year, now.month)[1]
    days_left = days_in - now.day + 1
    before_today = used_month - used_today
    daily = max(0, (eff - before_today)) / days_left
    return {"limit": limit, "effective_limit": eff, "used_month": used_month, "used_today": used_today,
            "days_left": days_left, "daily_share": daily, "remaining_today": max(0.0, daily - used_today),
            "remaining_month": max(0, eff - used_month)}


def can_spend(a: dict, cost: int) -> bool:
    return cost > 0 and cost <= a["remaining_today"] and cost <= a["remaining_month"]
