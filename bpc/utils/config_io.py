"""YAML-backed experiment configuration loading."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bpc.config import BPCConfig, LOGGING, LoggingConfig, make_presets


def load_yaml_like(path: str) -> Dict[str, Any]:
    text = Path(path).read_text()
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            value = _parse_scalar(stripped[2:].strip())
            parent = stack[-1][1]
            parent.setdefault("__list__", []).append(value)
            continue
        key, _, value = stripped.partition(":")
        while stack and indent < stack[-1][0]:
            finished = stack.pop()[1]
            if set(finished.keys()) == {"__list__"}:
                parent = stack[-1][1]
                for parent_key, parent_value in list(parent.items()):
                    if parent_value is finished:
                        parent[parent_key] = finished["__list__"]
                        break
        if stack and isinstance(stack[-1][1], dict) and set(stack[-1][1].keys()) == {"__list__"} and indent <= stack[-1][0]:
            finished = stack.pop()[1]
            parent = stack[-1][1]
            for parent_key, parent_value in list(parent.items()):
                if parent_value is finished:
                    parent[parent_key] = finished["__list__"]
                    break
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            node: Dict[str, Any] = {}
            current[key] = node
            stack.append((indent + 2, node))
        else:
            current[key] = _parse_scalar(value.strip())
    while len(stack) > 1:
        finished = stack.pop()[1]
        if isinstance(finished, dict) and set(finished.keys()) == {"__list__"}:
            parent = stack[-1][1]
            for parent_key, parent_value in list(parent.items()):
                if parent_value is finished:
                    parent[parent_key] = finished["__list__"]
                    break
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config_with_presets(
    path: str,
    default_cfg,
    presets: Dict[str, object],
    preset_override: Optional[str] = None,
) -> Tuple[object, LoggingConfig, Dict[str, Any]]:
    """Generic YAML + preset loader shared by the BPC and VBPC entry points."""

    raw = load_yaml_like(path)
    preset = preset_override or raw.get("preset")
    if preset:
        if preset not in presets:
            raise KeyError(f"Unknown preset={preset}; available presets: {sorted(presets)}")
        cfg = presets[preset]
    else:
        cfg = default_cfg
    cfg = _replace_known(cfg, raw.get("config", {}))
    lcfg = _replace_known(LOGGING, raw.get("logging", {}))
    return cfg, lcfg, raw


def load_experiment_config(
    path: str,
    default_cfg: BPCConfig,
    preset_override: Optional[str] = None,
) -> Tuple[BPCConfig, LoggingConfig, Dict[str, Any]]:
    return load_config_with_presets(path, default_cfg, make_presets(), preset_override)


def _replace_known(obj, values: Dict[str, Any]):
    allowed = {f.name for f in fields(obj)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config fields for {type(obj).__name__}: {unknown}")
    return replace(obj, **values)
