"""Deterministic hashing + the hash tree used for dependency-aware recompute (spec §59-60).

Every layer hash is derived from its own inputs AND the hashes of its parents, so a change
upstream invalidates exactly the downstream layers and nothing else.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

# Ordered dependency graph. Key -> parents.
LAYER_PARENTS: dict[str, tuple[str, ...]] = {
    "ROSTER": (),
    "INJURY": (),
    "STARTERS": (),
    "MODEL": (),                       # model versions + fitted-parameter digest
    "ROTATION": ("ROSTER", "INJURY", "STARTERS", "MODEL"),
    "MINUTES": ("ROTATION",),
    "OPPORTUNITY": ("MINUTES",),
    "DISTRIBUTION": ("OPPORTUNITY",),  # Monte Carlo output (Phase 2+)
    "MARKET": (),                      # sportsbook lines: never a parent of model layers (spec §28)
    "MARKET_COMPARISON": ("DISTRIBUTION", "MARKET"),
}
LAYER_ORDER = ["ROSTER", "INJURY", "STARTERS", "MODEL", "ROTATION", "MINUTES",
               "OPPORTUNITY", "DISTRIBUTION", "MARKET", "MARKET_COMPARISON"]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "value"):
        return o.value
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:n]


@dataclass
class HashTree:
    """Hashes per layer for one game. `own` = digest of the layer's direct inputs."""
    own: dict[str, str] = field(default_factory=dict)

    def set_input(self, layer: str, payload: Any) -> None:
        if layer not in LAYER_PARENTS:
            raise KeyError(layer)
        self.own[layer] = sha(payload)

    def resolved(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for layer in LAYER_ORDER:
            parents = LAYER_PARENTS[layer]
            out[layer] = sha({"own": self.own.get(layer, ""),
                              "parents": [out[p] for p in parents]})
        return out


def changed_layers(old: Mapping[str, str] | None, new: Mapping[str, str]) -> set[str]:
    if not old:
        return set(new)
    return {k for k, v in new.items() if old.get(k) != v}


def recompute_plan(old: Mapping[str, str] | None, new: Mapping[str, str]) -> list[str]:
    """Layers that must be recomputed, in execution order. Input layers (no compute) excluded."""
    compute_layers = {"ROTATION", "MINUTES", "OPPORTUNITY", "DISTRIBUTION", "MARKET_COMPARISON"}
    ch = changed_layers(old, new)
    return [l for l in LAYER_ORDER if l in ch and l in compute_layers]
