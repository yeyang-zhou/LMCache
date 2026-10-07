# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from lmcache.integration.vllm import lmcache_mp_metadata as metadata_mod
from lmcache.integration.vllm import token_drop
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
from lmcache.integration.vllm.token_drop import (
    TokenDropSpec,
    build_token_drop_algorithm,
    parse_token_drop_spec,
)


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_absent_token_drop_config_means_normal_request() -> None:
    assert parse_token_drop_spec(None) is None
    assert parse_token_drop_spec({}) is None
    assert parse_token_drop_spec({"lmcache.max_offload_tokens": 64}) is None


def test_parse_request_local_token_drop_config_is_opaque_to_lmcache() -> None:
    config = {
        "budget": 1024,
        "buffer": 64,
        "window_size": 8,
        "kernel_size": 7,
    }
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "rkv",
                "config": config,
            }
        }
    )

    assert spec == TokenDropSpec(
        algorithm="rkv",
        config=config,
    )


def test_algorithm_factory_uses_external_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def factory(config):
        seen.update(config)
        return SimpleNamespace(name="fake")

    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [_FakeEntryPoint("fake", factory)],
    )

    algorithm = build_token_drop_algorithm(
        TokenDropSpec(
            algorithm="fake",
            config={"opaque": 1},
        )
    )
    assert algorithm.name == "fake"
    assert seen == {"opaque": 1}


def test_missing_algorithm_plugin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: [])

    with pytest.raises(ValueError, match="No token-drop algorithm plugin"):
        build_token_drop_algorithm(TokenDropSpec("missing", {}))


def test_request_tracker_records_token_drop_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_configs = {
        "lmcache.token_drop": {
            "algorithm": "rkv",
            "config": {"budget": 1024, "buffer": 64},
        }
    }
    monkeypatch.setattr(
        metadata_mod,
        "extract_request_configs_from_request",
        lambda _: request_configs,
    )
    monkeypatch.setattr(metadata_mod, "extract_mm_features", lambda _: (None, None))

    request = SimpleNamespace(
        request_id="td",
        cache_salt="",
        all_token_ids=[],
        num_prompt_tokens=0,
    )
    tracker = LMCacheMPRequestTracker(request)

    assert tracker.token_drop_spec == TokenDropSpec(
        algorithm="rkv",
        config={"budget": 1024, "buffer": 64},
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("rkv", "JSON object"),
        ({"config": {"budget": 32, "buffer": 16}}, "algorithm"),
        ({"algorithm": "rkv"}, "config"),
        ({"algorithm": "rkv", "config": []}, "config"),
        (
            {
                "algorithm": "rkv",
                "config": {"budget": 32, "buffer": 16},
                "unexpected": True,
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_token_drop_envelope_fails_closed(raw, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_token_drop_spec({"lmcache.token_drop": raw})


def test_lmcache_does_not_validate_algorithm_specific_config() -> None:
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "rkv",
                "config": {
                    "budget": "validated-by-rkv",
                    "some_future_rkv_knob": object(),
                },
            }
        }
    )
    assert spec is not None
    assert spec.config["budget"] == "validated-by-rkv"
    assert "some_future_rkv_knob" in spec.config
