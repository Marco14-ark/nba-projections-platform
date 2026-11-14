"""Real training frame (2024-25 + 2025-26 regular season), LEAK-FREE absences (v2)."""
import sys, pandas as pd
sys.path.insert(0, '/home/claude/nba-platform')
from nba_platform.sources.pbp_reconstruct import infer_absences_asof
D = '/home/claude/data'
games = pd.concat([pd.read_pickle(f'{D}/cdnnba_{s}_games.pkl') for s in ('2024', '2025')], ignore_index=True)
logs = pd.concat([pd.read_pickle(f'{D}/cdnnba_{s}_logs.pkl') for s in ('2024', '2025')], ignore_index=True)
full = infer_absences_asof(logs, games)
full.to_pickle(f'{D}/real_logs_v2.pkl'); games.to_pickle(f'{D}/real_games_v2.pkl')
print(len(games), len(logs), len(full), full.not_playing_reason.value_counts(dropna=False).to_dict(),
      'flagged games', (games['flags'].fillna('') != '').sum())
