"""V10 props + game markets walk-forward: Champion STACK_V2_ENV vs challengers on identical games/seeds.
Args: NAME START END STEP -> docs/validation/v10/st_NAME_START_STEP_{props,games}.pkl"""
import json, os, sys, time
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.sim.game_sim import JointConfig
from nba_platform.sim.rates import RatesConfig
from nba_platform.sim.validate import walk_forward, MARKETS

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
BASE_R = dict(hl_player_games=45.0, league_env=True)
CFG = {
    "CHAMPION": ({}, {}, {}),
    "MIN_V2_ROLE": ({"start_mix": True}, {}, {}),
    "MIN_V2_RETURN": ({"return_model": "role_decay"}, {}, {}),
    "MIN_V2_JOINT": ({"proj_weight": "var_role"}, {}, {}),
    "MIN_V2_ALL": ({"start_mix": True, "return_model": "role_decay", "proj_weight": "var_role"}, {}, {}),
    "SHOT_V2_USAGE": ({}, {"hl_use_games": 15.0}, {}),
    "JOINT_MC_V2": ({}, {}, {"form_sd": 0.1268, "team_use_sd": 0.0403, "ast_on_makes": True}),
    "JOINT_MC_V2_TEAM": ({}, {}, {"team_use_sd": 0.0403}),
    "STACK_V5_PACE": ({}, {"pace_norm": "team_mean"}, {}),
}
name, start, end, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
mc, rc, jc = CFG[name]
logs = pd.read_pickle(f"{D}/real_logs_v3.pkl"); games = pd.read_pickle(f"{D}/real_games_v3.pkl")
seg = pd.concat([pd.read_pickle(f"{D}/cdnnba_{s}_segments.pkl") for s in ("2024", "2025", "2026")])
games = games.merge(pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]], on="game_id", how="left")
bad = set(games.loc[games["flags"].fillna("") != "", "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[::step]
mk = {**MARKETS, "FGA": ["fga"], "3PA": ["fg3a"], "FTA": ["fta"], "FGM": ["fgm"]}
t = time.time()
P, G = walk_forward(logs, games, dates, n_sims=4000, anchor="rating", minutes_cfg=MinutesConfig(**mc),
                    rates_cfg=RatesConfig(**BASE_R, **rc), rating_blend=(0.295, 0.845), segments=seg,
                    joint=JointConfig(**jc) if jc else None, markets=mk)
P = P[~P.game_id.isin(bad)]; G = G[~G.game_id.isin(bad)]
out = os.path.join(ROOT, "docs/validation/v10")
os.makedirs(out, exist_ok=True)
P.to_pickle(f"{out}/st_{name}_{start}_{step}_props.pkl"); G.to_pickle(f"{out}/st_{name}_{start}_{step}_games.pkl")
print(json.dumps({"name": name, "start": start, "games": len(G), "s": round(time.time() - t)}), flush=True)
