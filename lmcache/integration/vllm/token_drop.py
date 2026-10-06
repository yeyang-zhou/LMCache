# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration for the vLLM integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenDropSpec:
    """Generic per-request token-dropping envelope."""

    algorithm: str
    config: dict[str, Any]


def parse_token_drop_spec(
    request_configs: Mapping[str, Any] | None,
) -> TokenDropSpec | None:
    """Parse lmcache.token_drop; absence means vanilla request behavior."""
    if not request_configs or "lmcache.token_drop" not in request_configs:
        return None

    raw = request_configs["lmcache.token_drop"]
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.token_drop must be a JSON object")

    allowed = {"algorithm", "config"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported lmcache.token_drop keys: {sorted(unknown)}")

    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("lmcache.token_drop.algorithm must be a non-empty string")

    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("lmcache.token_drop.config must be a JSON object")

    return TokenDropSpec(
        algorithm=algorithm,
        config=dict(config),
    )
