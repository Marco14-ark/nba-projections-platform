"""Time helpers. NBA 'game date' is the US/Eastern calendar date of tip-off (spec §55)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def parse_utc(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def game_date_et(tip_utc: datetime) -> date:
    return tip_utc.astimezone(ET).date()


def iso_duration_minutes(s: str | None) -> float:
    """NBA live boxscore minutes look like 'PT34M12.00S'. Returns decimal minutes."""
    if not s:
        return 0.0
    s = s.strip()
    if s.startswith("PT"):
        body = s[2:]
        mins = 0.0
        if "M" in body:
            m, body = body.split("M", 1)
            mins += float(m or 0)
        if body.endswith("S"):
            mins += float(body[:-1] or 0) / 60.0
        return mins
    if ":" in s:  # 'MM:SS'
        m, sec = s.split(":", 1)
        return float(m) + float(sec) / 60.0
    return float(s)


def season_for_date(d: date) -> str:
    """'2026-27' for any date from Aug 2026 to Jul 2027."""
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def minutes_before(tip_utc: datetime, now: datetime) -> float:
    return (tip_utc - now) / timedelta(minutes=1)
