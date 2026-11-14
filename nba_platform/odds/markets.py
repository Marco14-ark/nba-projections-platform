"""Canonical market codes and provider mappings."""
from __future__ import annotations

PLAYER_MARKETS = ("PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "FGA", "FGM", "3PA")
GAME_MARKETS = ("ML", "SPREAD", "TOTAL", "TEAM_TOTAL")

# SportsGameOdds statID -> canonical player market
SGO_PLAYER_STATS = {
    "points": "PTS", "rebounds": "REB", "assists": "AST", "threePointersMade": "3PM",
    "points+rebounds+assists": "PRA", "points+rebounds": "PR", "points+assists": "PA",
    "rebounds+assists": "RA", "fieldGoalsAttempted": "FGA", "fieldGoalsMade": "FGM",
    "threePointersAttempted": "3PA",
}

# The Odds API market key -> canonical. 'player_field_goals' is documented only as "Field Goals
# (Over/Under)"; it is mapped to FGM (made) and flagged so it can be audited. No FGA / 3PA keys exist.
TOA_PLAYER_MARKETS = {
    "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "3PM",
    "player_points_rebounds_assists": "PRA", "player_points_rebounds": "PR", "player_points_assists": "PA",
    "player_rebounds_assists": "RA", "player_field_goals": "FGM",
}
TOA_GAME_MARKETS = {"h2h": "ML", "spreads": "SPREAD", "totals": "TOTAL", "team_totals": "TEAM_TOTAL"}

# Markets the model can price from its stored distributions
MODEL_PRICED = set(PLAYER_MARKETS) | set(GAME_MARKETS)
