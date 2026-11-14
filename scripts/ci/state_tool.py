"""State branch helpers.
  restore DIR          reassemble <db>.gz.partNN chunks into <db> (and delete the chunks)
  baseline DIR OUT     record row counts of every table of every *.db
  verify DIR BASE      fail (exit 1) if any DB vanished, fails quick_check, or lost rows (append-only)
  pack DIR             gzip every *.db into 45 MB chunks <db>.gz.partNN and remove the raw *.db
GitHub rejects files > 100 MB; chunking keeps every DB pushable however large it grows."""
import glob, gzip, json, os, shutil, sqlite3, sys

CHUNK = 45 * 1024 * 1024


def dbs(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.db"), recursive=True))


def restore(d):
    heads = sorted(glob.glob(os.path.join(d, "**", "*.db.gz.part00"), recursive=True))
    for h in heads:
        base = h[:-len(".gz.part00")]
        parts = sorted(glob.glob(base + ".gz.part*"))
        tmp = base + ".gz"
        with open(tmp, "wb") as w:
            for p in parts:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
        with gzip.open(tmp, "rb") as r, open(base + ".tmp", "wb") as w:
            shutil.copyfileobj(r, w)
        os.replace(base + ".tmp", base)
        for p in parts + [tmp]:
            os.remove(p)
        print("restored", os.path.relpath(base, d), os.path.getsize(base))


def counts(path):
    c = sqlite3.connect(path)
    out = {t: c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
           for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    ok = c.execute("PRAGMA quick_check").fetchone()[0]
    c.close()
    return out, ok


def baseline(d, out):
    json.dump({os.path.relpath(p, d): counts(p)[0] for p in dbs(d)}, open(out, "w"), indent=1)


def verify(d, base):
    b = json.load(open(base))
    bad = []
    for rel, tabs in b.items():
        p = os.path.join(d, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: missing")
            continue
        now, ok = counts(p)
        if ok != "ok":
            bad.append(f"{rel}: quick_check {ok}")
        for t, n in tabs.items():
            if now.get(t, 0) < n:
                bad.append(f"{rel}.{t}: {n} -> {now.get(t, 0)} rows")
    print(json.dumps({"verified": len(b), "problems": bad}))
    sys.exit(1 if bad else 0)


def pack(d):
    for p in dbs(d):
        with open(p, "rb") as r, gzip.open(p + ".gz", "wb", compresslevel=6) as w:
            shutil.copyfileobj(r, w)
        i = 0
        with open(p + ".gz", "rb") as r:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                with open(f"{p}.gz.part{i:02d}", "wb") as w:
                    w.write(b)
                i += 1
        os.remove(p + ".gz")
        os.remove(p)
        print("packed", os.path.relpath(p, d), i, "chunk(s)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"restore": lambda: restore(sys.argv[2]), "baseline": lambda: baseline(sys.argv[2], sys.argv[3]),
     "verify": lambda: verify(sys.argv[2], sys.argv[3]), "pack": lambda: pack(sys.argv[2])}[cmd]()
