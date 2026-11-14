"""Free historical source: github.com/shufinskiy/nba_data (Apache-2.0).

Season archives of the NBA CDN play-by-play (`cdnnba_<startyear>.tar.xz`, regular season;
`cdnnba_po_<startyear>` playoffs). Downloaded with plain git over HTTPS (no API, no key, no cost),
fetching only the requested blobs.

As verified on 2026-11-13: complete 2024-25 and 2025-26 regular seasons available; no 2026-27
NBA file yet (the repo is updated periodically). For the live 2026-27 season use
`pipeline.ingest` against cdn.nba.com from a machine with access (see docs: free runtime options).
"""
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pandas as pd

from ..db.store import Store
from ..roster.sync import RosterEntry, RosterSync
from ..sources.teams import TEAMS
from .pbp_reconstruct import infer_absences, reconstruct_season

REPO = "https://github.com/shufinskiy/nba_data"


def fetch(names: list[str], workdir: str | Path) -> list[Path]:
    """git partial clone + checkout of just the named archives (e.g. ['cdnnba_2025'])."""
    wd = Path(workdir)
    repo = wd / "nba_data"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, str(repo)],
                       check=True)
    paths = [f"datasets/{n}.tar.xz" for n in names]
    subprocess.run(["git", "-C", str(repo), "checkout", "HEAD", "--", *paths], check=True)
    out = []
    for p in paths:
        with tarfile.open(repo / p) as t:
            t.extractall(wd, filter="data")
            out += [wd / m.name for m in t.getmembers() if m.name.endswith(".csv")]
    return out


def player_names(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["personId", "playerName", "playerNameI", "teamId"], low_memory=False)
    df = df[(df["personId"] > 0) & df["playerNameI"].notna()]
    return df.drop_duplicates("personId", keep="last")


def load_into_store(store: Store, csv_paths: list[str | Path]) -> dict:
    for t in TEAMS:
        store.upsert_team(*t)
    games, logs, names = [], [], []
    for p in csv_paths:
        g, l, _ = reconstruct_season(str(p))
        games.append(g)
        logs.append(l)
        names.append(player_names(p))
    G = pd.concat(games, ignore_index=True)
    L = infer_absences(pd.concat(logs, ignore_index=True), G)
    N = pd.concat(names).drop_duplicates("personId", keep="last")
    for r in N.itertuples(index=False):
        init = str(r.playerNameI)
        first = init.split(". ")[0] if ". " in init else ""
        store.x("""INSERT OR IGNORE INTO players(player_id, nba_person_id, full_name, first_name, last_name,
                   name_key, created_at) VALUES (?,?,?,?,?,?,datetime('now'))""",
                (f"P{int(r.personId)}", int(r.personId), init, first, str(r.playerName),
                 f"{str(r.playerName).lower()}|{first.lower()}"))
    for r in G.to_dict("records"):
        store.upsert_game(r)
    for r in L.to_dict("records"):
        store.upsert_log(r)
    # effective-dated rosters from appearances (first appearance per team = stint start)
    rs = RosterSync(store)
    app = L[L["played"] == 1].sort_values("game_date_et")
    events = 0
    for d, day in app.groupby("game_date_et"):
        ents = []
        for pid, t in day[["player_id", "team_id"]].drop_duplicates("player_id").itertuples(index=False):
            cur = store.q("SELECT team_id FROM roster_stints WHERE player_id=? AND end_date IS NULL", (pid,))
            if not cur or cur[0]["team_id"] != t:
                ents.append(RosterEntry(int(t), "", "", nba_person_id=int(pid[1:])))
        if ents:
            events += len(rs.ingest_snapshot("nba_pbp_appearance", d, ents, covered_team_ids=set()).events)
    store.commit()
    return {"games": len(G), "player_games": len(L), "players": len(N), "roster_events": events}
