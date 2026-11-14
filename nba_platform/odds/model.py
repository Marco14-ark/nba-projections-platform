"""Quote dataclass shared by providers."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class Quote:
    provider: str
    book: str
    provider_event_id: str
    home_name: str
    away_name: str
    commence_utc: str
    kind: str                 # PLAYER | GAME
    market: str               # PTS..3PA | ML | SPREAD | TOTAL | TEAM_TOTAL
    side: str                 # over | under | home | away
    price: int                # American odds
    line: float | None = None # None for ML; spread line is from the side's own perspective
    team_side: str | None = None        # for TEAM_TOTAL: home | away
    player_name: str | None = None
    provider_player_id: str | None = None
    book_updated_at: str | None = None
    available: bool = True
    notes: list = field(default_factory=list)

    def key(self) -> tuple:
        return (self.provider, self.book, self.provider_event_id, self.kind, self.market,
                self.team_side or "", self.provider_player_id or self.player_name or "", self.side)

    def dict(self) -> dict:
        return asdict(self)
