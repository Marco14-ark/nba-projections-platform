import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.sources.cdn_jsonl import iter_games, actions_frame, official_box
from nba_platform.sources.pbp_reconstruct import reconstruct_game
from nba_platform.core.timeutil import parse_utc, game_date_et, season_for_date
P = "/home/claude/data/s2627/nba_2026_27_pbp_box.jsonl.gz"
games, logs, box = [], [], []
for g in iter_games(P):
    r = reconstruct_game(actions_frame(g))
    tip = parse_utc(g["box"]["gameTimeUTC"]); d = game_date_et(tip).isoformat()
    games.append({"game_id": r.game_id, "season": "2026-27", "game_date_et": d, "tip_utc": tip.isoformat(),
                  "home_team_id": int(g["box"]["homeTeam"]["teamId"]), "away_team_id": int(g["box"]["awayTeam"]["teamId"]),
                  "home_score": int(g["box"]["homeTeam"]["score"]), "away_score": int(g["box"]["awayTeam"]["score"]),
                  "periods": int(g["box"]["period"]), "status": 3,
                  "home_poss": r.possessions.get(int(g["box"]["homeTeam"]["teamId"]), 0),
                  "away_poss": r.possessions.get(int(g["box"]["awayTeam"]["teamId"]), 0), "flags": ";".join(r.flags),
                  "recon_home": r.home_team_id})
    for pid, s in r.players.items():
        logs.append({"game_id": r.game_id, "player_id": f"P{pid}", "team_id": s["team_id"], "game_date_et": d,
                     "started": s["started"], "played": 1, "minutes": s["sec"] / 60, "not_playing_reason": None,
                     "reb": s["oreb"] + s["dreb"], **{k: v for k, v in s.items() if k not in ("team_id", "started")}})
    box += official_box(g)
G, L, B = pd.DataFrame(games), pd.DataFrame(logs), pd.DataFrame(box)
m = B.merge(L[["game_id", "player_id", "minutes", "started", "pts"]], on=["game_id", "player_id"], how="left")
pl = m[m.official_played == 1]
print("games", len(G), "flagged", (G["flags"] != "").sum(), "home mismatch", (G.recon_home != G.home_team_id).sum())
print("played players: official", len(pl), "found in recon", pl.minutes.notna().sum())
print("minutes |recon-official| mean", round((pl.minutes - pl.official_minutes).abs().mean(), 4), "max", round((pl.minutes - pl.official_minutes).abs().max(), 3), "share within 0.1 min", round(((pl.minutes - pl.official_minutes).abs() < 0.1).mean(), 4))
print("starters agree", round((pl.started == pl.official_starter).mean(), 4), "points agree", round((pl.pts == pl.official_pts).mean(), 4))
print("official not-playing reasons", B[B.official_played == 0].not_playing_reason.value_counts().head(8).to_dict())
G.to_pickle("/home/claude/data/s2627/games.pkl"); L.to_pickle("/home/claude/data/s2627/logs_recon.pkl"); B.to_pickle("/home/claude/data/s2627/box_official.pkl")
