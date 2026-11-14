"""Official NBA Injury Report (PDF) — URL discovery and parsing.

The league publishes PDFs at
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MM{AM|PM}.pdf
(timestamps in US/Eastern, 12-hour clock, currently every 15 minutes on game days).

Columns: Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason
Date/time/matchup/team cells are only printed on the first row of their group and must be
carried forward. Reason cells wrap and are vertically centred, so wrapped fragments can appear
above or below the row's baseline. `parse_pdf` uses word x/y positions to rebuild rows;
`parse_text_lines` is a fallback for plain extracted text.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.enums import AvailabilityStatus
from ..core.names import name_key_from_last_first
from ..core.timeutil import ET
from .teams import BY_FULL_NAME, team_id_from_full_name

BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{hh}_{mm}{ap}.pdf"

OFFICIAL_STATUSES = {"Available": AvailabilityStatus.AVAILABLE,
                     "Probable": AvailabilityStatus.PROBABLE,
                     "Questionable": AvailabilityStatus.QUESTIONABLE,
                     "Doubtful": AvailabilityStatus.DOUBTFUL,
                     "Out": AvailabilityStatus.OUT}
NYS = "NOT YET SUBMITTED"
COLS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]


def report_url(ts_et: datetime) -> str:
    ts = ts_et.astimezone(ET)
    hh = ts.strftime("%I")
    return BASE.format(d=ts.strftime("%Y-%m-%d"), hh=hh, mm=ts.strftime("%M"), ap=ts.strftime("%p"))


def candidate_urls(now_et: datetime, lookback_minutes: int = 180, step: int = 15) -> list[str]:
    """Newest-first list of plausible report URLs (fetch until the first 200)."""
    now = now_et.astimezone(ET)
    floored = now.replace(minute=(now.minute // step) * step, second=0, microsecond=0)
    return [report_url(floored - timedelta(minutes=k)) for k in range(0, lookback_minutes + 1, step)]


@dataclass
class InjuryRow:
    game_date: str           # MM/DD/YYYY
    game_time_et: str        # 'HH:MM (ET)'
    matchup: str             # 'AWY@HOM'
    team_name: str
    team_id: int | None
    player_raw: str | None   # 'Last, First' or None for NOT YET SUBMITTED
    name_key: str | None
    official_status: str | None
    status: AvailabilityStatus | None
    reason: str = ""
    not_yet_submitted: bool = False
    reason_parts: list[str] = field(default_factory=list, repr=False)


def _nospace(x: str) -> str:
    return "".join(x.split()).lower()


def classify(official: str, reason: str) -> AvailabilityStatus:
    """Map official status + reason to our enum without inventing anything.

    The official status is authoritative; the reason refines OUT into REST / SUSPENDED /
    NOT_WITH_TEAM, which matters for the minutes model (no injury-return ramp for rest).
    """
    st = OFFICIAL_STATUSES[official]
    r = _nospace(reason)       # the real PDF renders words without spaces ("GLeague-Two-Way")
    if st is AvailabilityStatus.OUT:
        if "suspension" in r:
            return AvailabilityStatus.SUSPENDED
        if r.startswith("rest") or "-rest" in r or r == "rest":
            return AvailabilityStatus.REST
        if r.startswith("gleague") or "notwithteam" in r:
            return AvailabilityStatus.NOT_WITH_TEAM
    return st


_DATE = r"\d{2}/\d{2}/\d{4}"
_TIME = r"\d{2}:\d{2} \(ET\)"
_MATCH = r"[A-Z]{2,3}@[A-Z]{2,3}"
_TEAMS_ALT = "|".join(sorted((re.escape(n) for n in BY_FULL_NAME), key=len, reverse=True))
_STATUS_ALT = "|".join(OFFICIAL_STATUSES)
ROW_RE = re.compile(
    rf"^(?:(?P<date>{_DATE})\s+)?(?:(?P<time>{_TIME})\s+)?(?:(?P<match>{_MATCH})\s+)?"
    rf"(?:(?P<team>{_TEAMS_ALT})\s+)?"
    rf"(?:(?P<nys>{NYS})|(?P<player>[^,]+?,\s*[^\s].*?)\s+(?P<status>{_STATUS_ALT})(?:\s+(?P<reason>.*))?)\s*$")
SKIP_RE = re.compile(r"^(Injury Report:|Page \d+ of \d+|Game Date\s+Game Time)")


def _finalise(rows: list[InjuryRow]) -> list[InjuryRow]:
    for r in rows:
        r.reason = " ".join(" ".join(r.reason_parts).split())
        if r.official_status:
            r.status = classify(r.official_status, r.reason)
    return rows


def parse_text_lines(text: str) -> list[InjuryRow]:
    rows: list[InjuryRow] = []
    ctx = {"date": "", "time": "", "match": "", "team": ""}
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line or SKIP_RE.match(line):
            continue
        m = ROW_RE.match(line)
        if not m:
            if rows and rows[-1].player_raw:
                rows[-1].reason_parts.append(line)  # wrapped reason continuation
            continue
        for k in ("date", "time", "match", "team"):
            if m.group(k):
                ctx[k] = m.group(k)
                if k in ("date", "time"):
                    # a new date/time block always comes with its own matchup+team
                    pass
        if m.group("nys"):
            rows.append(InjuryRow(ctx["date"], ctx["time"], ctx["match"], ctx["team"],
                                  team_id_from_full_name(ctx["team"]), None, None, None, None,
                                  not_yet_submitted=True))
            continue
        player = m.group("player").strip()
        r = InjuryRow(ctx["date"], ctx["time"], ctx["match"], ctx["team"],
                      team_id_from_full_name(ctx["team"]), player, name_key_from_last_first(player),
                      m.group("status"), None)
        if m.group("reason"):
            r.reason_parts.append(m.group("reason"))
        rows.append(r)
    return _finalise(rows)


# ---------------------------------------------------------------------------------------------
# Positional PDF parser
# ---------------------------------------------------------------------------------------------
HEADER_TOKENS = [("game_date", "gamedate"), ("game_time", "gametime"), ("matchup", "matchup"), ("team", "team"),
                 ("player", "playername"), ("status", "currentstatus"), ("reason", "reason")]


def _column_starts(words: list[dict]) -> dict[str, float] | None:
    """Header detection tolerant to both 'Game Date' and 'GameDate' renderings."""
    seq = [_nospace(w["text"]) for w in words]
    starts: dict[str, float] = {}
    i = 0
    for col, target in HEADER_TOKENS:
        found = False
        while i < len(seq):
            if seq[i] == target:
                starts[col], i, found = words[i]["x0"], i + 1, True
                break
            if i + 1 < len(seq) and seq[i] + seq[i + 1] == target:
                starts[col], i, found = words[i]["x0"], i + 2, True
                break
            i += 1
        if not found:
            return None
    return starts


def _lines(words: list[dict], tol: float = 2.5) -> list[list[dict]]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def parse_pdf(pdf_bytes: bytes) -> list[InjuryRow]:
    import pdfplumber  # local import: heavy

    rows: list[InjuryRow] = []
    ctx = {c: "" for c in ("game_date", "game_time", "matchup", "team")}
    starts: dict[str, float] | None = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            page_starts = _column_starts(words)
            starts = page_starts or starts
            if not starts:
                continue
            bounds = sorted(starts.items(), key=lambda kv: kv[1])
            page_rows: list[tuple[float, InjuryRow]] = []
            orphans: list[tuple[float, str]] = []
            for line in _lines(words):
                txt = " ".join(w["text"] for w in line)
                if SKIP_RE.match(txt) or _nospace(txt).startswith("gamedate") or _nospace(txt).startswith("injuryreport:") \
                        or re.match(r"^page\d+of\d+$", _nospace(txt)):
                    continue
                cells = {c: [] for c in COLS}
                for w in line:
                    col = bounds[0][0]
                    for c, x in bounds:
                        if w["x0"] >= x - 1.0:
                            col = c
                    cells[col].append(w["text"])
                cell = {c: " ".join(v) for c, v in cells.items()}
                y = line[0]["top"]
                for c in ("game_date", "game_time", "matchup", "team"):
                    if cell[c]:
                        ctx[c] = cell[c]
                if "notyetsubmitted" in _nospace(txt):
                    page_rows.append((y, InjuryRow(ctx["game_date"], ctx["game_time"], ctx["matchup"],
                                                   ctx["team"], team_id_from_full_name(ctx["team"]),
                                                   None, None, None, None, not_yet_submitted=True)))
                    continue
                stt = cell["status"].strip().title()
                if stt in OFFICIAL_STATUSES and cell["player"]:
                    cell["status"] = stt
                    r = InjuryRow(ctx["game_date"], ctx["game_time"], ctx["matchup"], ctx["team"],
                                  team_id_from_full_name(ctx["team"]), cell["player"],
                                  name_key_from_last_first(cell["player"]), cell["status"], None)
                    if cell["reason"]:
                        r.reason_parts.append(cell["reason"])
                    page_rows.append((y, r))
                elif cell["reason"] and not any(cell[c] for c in ("player", "status")):
                    orphans.append((y, cell["reason"]))
            # attach wrapped reason fragments to the vertically nearest player row
            anchors = [(y, r) for y, r in page_rows if r.player_raw]
            for y, frag in orphans:
                if not anchors:
                    continue
                ay, ar = min(anchors, key=lambda a: abs(a[0] - y))
                if y < ay:
                    ar.reason_parts.insert(0, frag)
                else:
                    ar.reason_parts.append(frag)
            rows.extend(r for _, r in page_rows)
    return _finalise(rows)
