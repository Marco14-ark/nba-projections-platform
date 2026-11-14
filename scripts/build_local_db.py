"""Build data/nba.db (SQLite, free) from the open PBP dataset: python scripts/build_local_db.py"""
import sys, time
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.db.store import Store
from nba_platform.sources.open_dataset import load_into_store
t = time.time()
s = Store("/home/claude/data/nba.db")
print(load_into_store(s, ["/home/claude/data/raw/cdnnba_2024.csv", "/home/claude/data/raw/cdnnba_2025.csv"]), round(time.time() - t), "s")
