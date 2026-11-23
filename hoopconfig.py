"""Configuration loading. `config.json` holds every tunable threshold; missing keys fall back to
the defaults shipped in the repository so old config files keep working."""
from __future__ import annotations

import copy
import json
import os
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(ROOT, "config.json")

_DEFAULTS: dict[str, Any] | None = None


def _defaults() -> dict[str, Any]:
    global _DEFAULTS
    if _DEFAULTS is None:
        with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
            _DEFAULTS = json.load(f)
    return copy.deepcopy(_DEFAULTS)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = _defaults()
    if path and os.path.abspath(path) != DEFAULT_PATH and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, json.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
