"""Source adapter tests. Fixtures reproduce the documented STRUCTURE of each feed; player names
and values in them are test data, not real reports."""
import io
from datetime import datetime

import pytest

from nba_platform.core.enums import AvailabilityStatus as S
from nba_platform.core.names import name_key, name_key_from_last_first
from nba_platform.core.timeutil import ET, iso_duration_minutes
from nba_platform.sources import nba_cdn
from nba_platform.sources.injury_report import candidate_urls, parse_pdf, parse_text_lines, report_url

REPORT_TEXT = """Injury Report: 01/27/26 10:45 PM
Page 1 of 3
Game Date Game Time Matchup Team Player Name Current Status Reason
01/27/2026 07:00 (ET) POR@WAS Portland Trail Blazers NOT YET SUBMITTED
Washington Wizards Alpha, Andre Out Injury/Illness - Right Knee; MCL
Strain; Quad Contusion
Beta Jr., Kelly Available Injury/Illness - Left Knee; Injury
Management
Gamma, Chris Out G League - Two-Way
07:30 (ET) DEN@BKN Denver Nuggets Jokić, Nikolai Questionable Injury/Illness - Left Knee; Bone Bruise
Delta, Dan Out Rest
Brooklyn Nets Epsilon, Eli Doubtful Injury/Illness - N/a; Illness
Zeta, Zed Out League Suspension
Injury Report: 01/27/26 10:45 PM
Page 2 of 3
09:30 (ET) SAS@HOU San Antonio Spurs NOT YET SUBMITTED
Houston Rockets NOT YET SUBMITTED
"""


def test_report_url_format_matches_league_naming():
    assert report_url(datetime(2026, 4, 26, 10, 30, tzinfo=ET)).endswith("Injury-Report_2026-04-26_10_30AM.pdf")
    assert report_url(datetime(2026, 4, 29, 0, 0, tzinfo=ET)).endswith("Injury-Report_2026-04-29_12_00AM.pdf")
    assert report_url(datetime(2026, 5, 5, 14, 15, tzinfo=ET)).endswith("Injury-Report_2026-05-05_02_15PM.pdf")
    urls = candidate_urls(datetime(2026, 11, 13, 18, 7, tzinfo=ET), lookback_minutes=30)
    assert urls[0].endswith("2026-11-13_06_00PM.pdf") and len(urls) == 3


def test_text_parser_carry_forward_wrapping_and_classification():
    rows = parse_text_lines(REPORT_TEXT)
    players = [r for r in rows if r.player_raw]
    nys = [r for r in rows if r.not_yet_submitted]
    assert {r.team_name for r in nys} == {"Portland Trail Blazers", "San Antonio Spurs", "Houston Rockets"}
    a = next(r for r in players if r.player_raw.startswith("Alpha"))
    assert a.reason == "Injury/Illness - Right Knee; MCL Strain; Quad Contusion"
    assert a.team_name == "Washington Wizards" and a.matchup == "POR@WAS" and a.game_date == "01/27/2026"
    b = next(r for r in players if r.player_raw.startswith("Beta"))
    assert b.status is S.AVAILABLE and b.name_key == "beta|kelly" and b.reason.endswith("Injury Management")
    st = {r.player_raw.split(",")[0]: r.status for r in players}
    assert st["Gamma"] is S.NOT_WITH_TEAM
    assert st["Delta"] is S.REST
    assert st["Zeta"] is S.SUSPENDED
    assert st["Epsilon"] is S.DOUBTFUL
    j = next(r for r in players if r.player_raw.startswith("Jok"))
    assert j.status is S.QUESTIONABLE and j.name_key == "jokic|nikolai" and j.matchup == "DEN@BKN"
    assert j.team_name == "Denver Nuggets"
    eps = next(r for r in players if r.player_raw.startswith("Epsilon"))
    assert eps.team_name == "Brooklyn Nets" and eps.game_time_et == "07:30 (ET)"


def _make_pdf() -> bytes:
    """Builds a PDF with the league's column layout, including a reason wrapped ABOVE the row
    baseline (vertical centring) to exercise the positional parser."""
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    c.setFont("Helvetica", 8)
    xs = [20, 90, 150, 200, 330, 450, 520]
    y = 560
    c.drawString(20, 585, "Injury Report: 02/27/26 02:45 PM")
    for x, t in zip(xs, ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]):
        c.drawString(x, y, t)
    y -= 16
    for x, t in zip(xs, ["02/27/2026", "07:00 (ET)", "CLE@DET", "Cleveland Cavaliers", "Ellis, Kyle", "Questionable"]):
        c.drawString(x, y, t)
    c.drawString(xs[6], y + 4, "Injury/Illness - Left Index Finger;")
    c.drawString(xs[6], y - 5, "Fracture")
    y -= 16
    c.drawString(xs[4], y, "Strand, Julian"); c.drawString(xs[5], y, "Probable")
    c.drawString(xs[6], y, "Injury/Illness - Left Great Toe; Sprain")
    y -= 16
    c.drawString(xs[6], y + 4, "Injury/Illness - Right Hamstring;")
    c.drawString(xs[3], y, "Detroit Pistons"); c.drawString(xs[4], y, "Watts, Peyton"); c.drawString(xs[5], y, "Out")
    c.drawString(xs[6], y - 5, "Strain")
    y -= 16
    for x, t in zip(xs[1:], ["09:30 (ET)", "HOU@MIA", "Houston Rockets", "NOT YET SUBMITTED"]):
        c.drawString(x, y, t)
    c.save()
    return buf.getvalue()


def test_pdf_positional_parser():
    rows = parse_pdf(_make_pdf())
    players = {r.player_raw: r for r in rows if r.player_raw}
    assert set(players) == {"Ellis, Kyle", "Strand, Julian", "Watts, Peyton"}
    assert players["Ellis, Kyle"].reason == "Injury/Illness - Left Index Finger; Fracture"
    assert players["Watts, Peyton"].reason == "Injury/Illness - Right Hamstring; Strain"
    assert players["Watts, Peyton"].team_name == "Detroit Pistons" and players["Watts, Peyton"].status is S.OUT
    assert players["Strand, Julian"].team_name == "Cleveland Cavaliers"
    nys = [r for r in rows if r.not_yet_submitted]
    assert len(nys) == 1 and nys[0].matchup == "HOU@MIA" and nys[0].team_id == 1610612745


def test_name_normalisation():
    assert name_key_from_last_first("Oubre Jr., Kelly") == name_key("Kelly", "Oubre Jr.") == "oubre|kelly"
    assert name_key("Nikola", "Jokić") == "jokic|nikola"
    assert name_key("Shai", "Gilgeous-Alexander") == "gilgeousalexander|shai"


BOX = {"game": {"gameId": "0022600123", "gameTimeUTC": "2026-11-14T00:30:00Z", "gameStatus": 3, "period": 5,
                "homeTeam": {"teamId": 1610612747, "teamTricode": "LAL", "score": 120, "players": [
                    {"status": "ACTIVE", "personId": 1, "firstName": "A", "familyName": "One", "starter": "1",
                     "played": "1", "statistics": {"minutes": "PT40M30.00S", "points": 30, "reboundsTotal": 5}},
                    {"status": "ACTIVE", "personId": 2, "firstName": "B", "familyName": "Two", "starter": "0",
                     "played": "1", "statistics": {"minutes": "PT224M30.00S", "points": 10}},
                    {"status": "INACTIVE", "personId": 3, "firstName": "C", "familyName": "Three", "starter": "0",
                     "played": "0", "notPlayingReason": "INACTIVE_INJURY", "statistics": {}}]},
                "awayTeam": {"teamId": 1610612738, "teamTricode": "BOS", "score": 118, "players": [
                    {"status": "ACTIVE", "personId": 4, "firstName": "D", "familyName": "Four", "starter": "1",
                     "played": "1", "statistics": {"minutes": "PT265M00.00S"}}]}}}


def test_boxscore_parse_and_team_accounting():
    bs = nba_cdn.parse_boxscore(BOX)
    assert iso_duration_minutes("PT34M12.00S") == pytest.approx(34.2)
    assert len(bs.players) == 4 and bs.players[0].starter and bs.players[0].minutes == pytest.approx(40.5)
    chk = nba_cdn.boxscore_minutes_check(bs)                # one OT -> 265 expected
    assert chk[1610612747] == pytest.approx(0.0) and chk[1610612738] == pytest.approx(0.0)
    rows = nba_cdn.boxscore_to_logs(bs, lambda p: f"P{p.person_id}")
    assert rows[2]["not_playing_reason"] == "INACTIVE_INJURY" and rows[0]["game_date_et"] == "2026-11-13"


def test_boxscore_refuses_non_final_and_detects_drift():
    live = {"game": {**BOX["game"], "gameStatus": 2}}
    with pytest.raises(ValueError):
        nba_cdn.boxscore_to_logs(nba_cdn.parse_boxscore(live), lambda p: "x")
    with pytest.raises(nba_cdn.SchemaDrift):
        nba_cdn.parse_boxscore({"game": {"gameId": "1"}})


def test_schedule_filters_preseason_and_tbd():
    doc = {"leagueSchedule": {"gameDates": [{"games": [
        {"gameId": "0012600001", "gameDateTimeUTC": "2026-10-03T23:00:00Z", "homeTeam": {"teamId": 1610612747}, "awayTeam": {"teamId": 1610612738}},
        {"gameId": "0022600001", "gameDateTimeUTC": "2026-10-21T02:30:00Z", "homeTeam": {"teamId": 1610612747, "teamTricode": "LAL"}, "awayTeam": {"teamId": 1610612738, "teamTricode": "BOS"}},
        {"gameId": "0042600101", "gameDateTimeUTC": "2027-04-18T19:00:00Z", "homeTeam": {"teamId": 0}, "awayTeam": {"teamId": 0}}]}]}}
    g = nba_cdn.parse_schedule(doc)
    assert [x.game_id for x in g] == ["0022600001"]
    assert g[0].to_row()["game_date_et"] == "2026-10-20"      # 02:30 UTC = previous evening ET
    assert g[0].to_row()["season"] == "2026-27"
