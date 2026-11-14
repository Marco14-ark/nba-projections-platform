"""Player-name normalisation for cross-source matching (injury PDF, odds feeds, third parties)."""
from __future__ import annotations

import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _clean(tok: str) -> str:
    tok = strip_accents(tok).lower()
    tok = re.sub(r"[^a-z0-9 ]", "", tok.replace("-", " "))
    parts = [p for p in tok.split() if p not in SUFFIXES]
    return "".join(parts)


def name_key(first: str, last: str) -> str:
    return f"{_clean(last)}|{_clean(first)}"


_SUFFIX_GLUED = re.compile(r"(?<=[a-z])(Jr\.?|Sr\.?|III|II|IV)$")


def name_key_from_last_first(s: str) -> str:
    """'Oubre Jr., Kelly' -> 'oubre|kelly'. 'Jokic, Nikola' -> 'jokic|nikola'.
    The real PDF glues suffixes: 'HolmesII,DaRon' -> 'holmes|daron'."""
    if "," in s:
        last, first = s.split(",", 1)
        last = _SUFFIX_GLUED.sub("", last.strip())
    else:
        parts = s.split()
        first, last = parts[0], " ".join(parts[1:])
    return name_key(first.strip(), last.strip())


def name_key_from_full(s: str) -> str:
    parts = [p for p in s.split() if _clean(p) and _clean(p) not in SUFFIXES] or s.split()
    return name_key(parts[0], " ".join(parts[1:]))
