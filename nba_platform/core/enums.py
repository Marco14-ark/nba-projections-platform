"""Shared enumerations. Values are stored as strings in the database."""
from __future__ import annotations

from enum import Enum


class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"
    SUSPENDED = "SUSPENDED"
    REST = "REST"
    GTD = "GTD"
    MINUTES_RESTRICTION = "MINUTES_RESTRICTION"
    NOT_WITH_TEAM = "NOT_WITH_TEAM"       # G League assignment / personal reasons
    UNKNOWN = "UNKNOWN"

    @property
    def is_definitely_out(self) -> bool:
        return self in (AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED,
                        AvailabilityStatus.REST, AvailabilityStatus.NOT_WITH_TEAM)

    @property
    def is_uncertain(self) -> bool:
        return self in (AvailabilityStatus.QUESTIONABLE, AvailabilityStatus.DOUBTFUL,
                        AvailabilityStatus.GTD)


class StatusOrigin(str, Enum):
    """Distinguishes official designations from model inferences (spec §5, §80)."""
    OFFICIAL = "OFFICIAL"          # NBA official injury report / team announcement via licensed feed
    LICENSED_FEED = "LICENSED_FEED"  # third-party licensed provider (e.g. balldontlie, SportsDataIO)
    MODEL_INFERENCE = "MODEL_INFERENCE"


class LineupState(str, Enum):
    EXPECTED = "EXPECTED"
    CONFIRMED = "CONFIRMED"
    UNAVAILABLE = "UNAVAILABLE"


class ForecastFlag(str, Enum):
    OK = "OK"
    PARTIAL_DATA = "PARTIAL_DATA"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RefreshWindow(str, Enum):
    INITIAL = "INITIAL"
    T_6H = "6H"
    T_90M = "90M"
    T_30M = "30M"
    T_15M = "15M"


class TransactionType(str, Enum):
    TRADE = "TRADE"
    SIGNING = "SIGNING"
    WAIVER = "WAIVER"
    RELEASE = "RELEASE"
    TWO_WAY = "TWO_WAY"
    GLEAGUE_ASSIGNMENT = "GLEAGUE_ASSIGNMENT"
    GLEAGUE_RECALL = "GLEAGUE_RECALL"
    SUSPENSION = "SUSPENSION"
    TEAM_CHANGE_DETECTED = "TEAM_CHANGE_DETECTED"   # inferred from roster diff, no explicit transaction
    ROSTER_ADD_DETECTED = "ROSTER_ADD_DETECTED"
    ROSTER_DROP_DETECTED = "ROSTER_DROP_DETECTED"
