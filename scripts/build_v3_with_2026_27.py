"""Dataset v3 = leak-free v2 (2024-26) + 2026-27 with OFFICIAL availability from boxscores."""
import sys, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
D = "/home/claude/data"
L2, G2 = pd.read_pickle(f"{D}/real_logs_v2.pkl"), pd.read_pickle(f"{D}/real_games_v2.pkl")
G3 = pd.read_pickle(f"{D}/s2627/games.pkl").drop(columns=["recon_home"])
R3 = pd.read_pickle(f"{D}/s2627/logs_recon.pkl")
B = pd.read_pickle(f"{D}/s2627/box_official.pkl")
np_ = B[B.official_played == 0].copy()
def lab(r):
    s = str(r or "")
    if s.startswith(("INACTIVE", "NWT")) or s in ("DND_INJURY", "DND_REST", "DND_ILLNESS", "DND_PERSONAL"):
        return "INACTIVE_OFFICIAL:" + s
    return "DNP_OFFICIAL:" + (s or "COACH")
np_["not_playing_reason"] = np_["not_playing_reason"].map(lab)
np_ = np_.assign(started=0, played=0, minutes=0.0)[["game_id", "player_id", "team_id", "game_date_et", "started", "played", "minutes", "not_playing_reason"]]
L3 = pd.concat([L2, R3, np_], ignore_index=True)
G = pd.concat([G2, G3], ignore_index=True)
L3.to_pickle(f"{D}/real_logs_v3.pkl"); G.to_pickle(f"{D}/real_games_v3.pkl")
print(len(G3), "games 2026-27;", len(R3), "played rows;", np_.not_playing_reason.str.split(":").str[0].value_counts().to_dict())
