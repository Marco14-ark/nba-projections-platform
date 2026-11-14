"""Props + game markets walk-forward on identical games/seeds for CHAMPION vs V2 stack.
Args: NAME START END STEP DATA(v2|v3)"""
import sys, json, time, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.sim.rates import RatesConfig
from nba_platform.sim.validate import walk_forward
D = "/home/claude/data"
name, start, end, step, ver = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
MCFG = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
logs = pd.read_pickle(f"{D}/real_logs_{ver}.pkl"); games = pd.read_pickle(f"{D}/real_games_{ver}.pkl")
seg = pd.concat([pd.read_pickle(f"{D}/cdnnba_{s}_segments.pkl") for s in ("2024", "2025", "2026")])
games = games.merge(pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]], on="game_id", how="left")
bad = set(games.loc[games["flags"].fillna("") != "", "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[::step]
kw = dict(n_sims=4000, anchor="rating")
if name == "CHAMPION":
    kw.update(minutes_cfg=MinutesConfig(**MCFG))   # pass {"dnp_model":"pooled"} for the 1.1.0 champion
if name == "STACK_V3_SIGMA":
    kw.update(minutes_cfg=MinutesConfig(**MCFG), rates_cfg=RatesConfig(hl_player_games=45.0, league_env=True),
              rating_blend=(0.295, 0.845), segments=seg)
if name == "STACK_V4_BLOWOUT":
    kw.update(minutes_cfg=MinutesConfig(**{"blowout_feature": "garbage", "dnp_garbage": True, **MCFG}),
              rates_cfg=RatesConfig(hl_player_games=45.0, league_env=True), rating_blend=(0.295, 0.845), segments=seg)
if name == "STACK_V2_ENV":
    kw.update(minutes_cfg=MinutesConfig(**MCFG), rates_cfg=RatesConfig(hl_player_games=45.0, league_env=True),
              rating_blend=(0.295, 0.845), segments=seg)
if name == "STACK_V2":
    kw.update(minutes_cfg=MinutesConfig(**MCFG), rates_cfg=RatesConfig(hl_player_games=45.0),
              rating_blend=(0.295, 0.845), segments=seg)
elif name == "RATING_ONLY":
    kw.update(rating_blend=(0.295, 0.845), segments=seg)
t = time.time()
P, G = walk_forward(logs, games, dates, **kw)
P = P[~P.game_id.isin(bad)]; G = G[~G.game_id.isin(bad)]
import os
SUF = os.environ.get("OUT_SUFFIX", "")
P.to_pickle(f"{D}/st_{name}{SUF}_{ver}_{start}_{end}_{step}_props.pkl"); G.to_pickle(f"{D}/st_{name}{SUF}_{ver}_{start}_{end}_{step}_games.pkl")
print(json.dumps({"name": name, "games": len(G), "s": round(time.time() - t)}), flush=True)
