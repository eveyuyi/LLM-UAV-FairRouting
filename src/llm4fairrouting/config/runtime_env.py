"""Helpers for loading CLI defaults from `.env` and environment variables."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _strip_inline_comment(value: str) -> str:
    if not value or value[0] in {"'", '"'}:
        return value
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def _normalize_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}

    with env_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                raise ValueError(f"Invalid .env line {line_number}: {raw_line.rstrip()}")

            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid .env line {line_number}: empty key")

            value = _normalize_value(raw_value)
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value

    return loaded


def _resolve_env_path(project_root: Path, candidate: str) -> Path:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_absolute():
        return candidate_path

    cwd_path = (Path.cwd() / candidate_path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (project_root / candidate_path).resolve()


def prepare_env_file(project_root: Path, argv: Sequence[str] | None = None) -> Path | None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(tokens)

    explicit_path = pre_args.env_file or os.environ.get("LLM4FAIRROUTING_ENV_FILE")
    if explicit_path:
        env_path = _resolve_env_path(project_root, explicit_path)
        if not env_path.exists():
            raise FileNotFoundError(f"未找到环境配置文件: {env_path}")
        load_env_file(env_path)
        return env_path

    default_path = project_root / ".env"
    if default_path.exists():
        load_env_file(default_path)
        return default_path

    return None


def env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def env_int(name: str, default: int) -> int:
    value = env_text(name)
    return default if value is None else int(value)


def env_optional_int(name: str, default: int | None = None) -> int | None:
    value = env_text(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = env_text(name)
    return default if value is None else float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_text(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Environment variable {name} must be one of {_TRUE_VALUES | _FALSE_VALUES}, got: {value}")


def env_int_list(name: str, default: Iterable[int] | None = None) -> list[int] | None:
    value = env_text(name)
    if value is None:
        return list(default) if default is not None else None

    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if not tokens:
        return []
    return [int(token) for token in tokens]
