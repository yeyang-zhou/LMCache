# SPDX-License-Identifier: Apache-2.0
"""Mandatory scheduler-to-worker token-drop metadata."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
)


@pytest.mark.parametrize(
    "required_field",
    ["physical_block_ids", "is_genuine_decode", "num_decoded_tokens", "num_new_tokens"],
)
def test_scheduler_must_explicitly_supply_request_state(required_field: str) -> None:
    values = dict(
        request_id="td",
        algorithm="rkv",
        config={},
        resident_kv_tokens=32,
        physical_block_ids=None,
        is_genuine_decode=False,
        num_decoded_tokens=0,
        num_new_tokens=8,
    )
    values.pop(required_field)
    with pytest.raises(TypeError):
        LMCacheMPTokenDropRequestState(**values)


def test_worker_row_is_the_only_default() -> None:
    state = LMCacheMPTokenDropRequestState(
        request_id="td",
        algorithm="rkv",
        config={},
        resident_kv_tokens=32,
        physical_block_ids=None,
        is_genuine_decode=False,
        num_decoded_tokens=0,
        num_new_tokens=8,
    )
    assert state.worker_row == -1
