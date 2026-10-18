# SPDX-License-Identifier: Apache-2.0
"""Scheduler/worker token-drop metadata and output barrier contracts."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop import TokenDropSpec
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.token_drop_allocator_adapter import (
    clear_resident_kv_tokens,
    set_resident_kv_tokens,
)


@pytest.fixture(autouse=True)
def _clear_frontier():
    clear_resident_kv_tokens("td")
    yield
    clear_resident_kv_tokens("td")


def _scheduler(*, logical=65, prompt=64):
    tracker = SimpleNamespace(
        token_drop_spec=TokenDropSpec("rkv", {"budget": 32}),
        num_scheduled_tokens=logical,
        num_prompt_tokens=prompt,
    )
    blocks = [
        SimpleNamespace(block_id=7, ref_cnt=1, block_hash=None),
        SimpleNamespace(block_id=9, ref_cnt=1, block_hash=None),
    ]
    scheduler = SimpleNamespace(
        request_trackers={
            "td": tracker,
            "normal": SimpleNamespace(token_drop_spec=None),
        },
        _require_private_token_drop_blocks=lambda req_id: blocks,
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["normal", "td"]),
        num_scheduled_tokens={"normal": 1, "td": 1},
    )
    return scheduler, output


def test_publish_private_resident_frontier_and_decode_facts():
    scheduler, output = _scheduler()
    set_resident_kv_tokens("td", 33)
    metadata = LMCacheMPConnectorMetadata()

    LMCacheMPConnector._add_token_drop_states(scheduler, output, metadata)

    assert len(metadata.token_drop_requests) == 1
    state = metadata.token_drop_requests[0]
    assert state.request_id == "td"
    assert state.resident_kv_tokens == 33
    assert state.has_physical_override is True
    assert state.physical_block_ids == [7, 9]
    assert state.private_blocks_validated is True
    assert state.is_genuine_decode is True
    assert state.num_decoded_tokens == 0
    assert state.num_new_tokens == 1
    assert len(metadata) == 1


def test_first_prefill_keeps_vanilla_physical_addressing():
    scheduler, output = _scheduler(logical=64, prompt=64)
    output.scheduled_cached_reqs.req_ids = []
    output.scheduled_new_reqs = [SimpleNamespace(req_id="td")]
    output.num_scheduled_tokens["td"] = 64
    metadata = LMCacheMPConnectorMetadata()

    LMCacheMPConnector._add_token_drop_states(scheduler, output, metadata)

    state = metadata.token_drop_requests[0]
    assert state.has_physical_override is False
    assert state.physical_block_ids is None
    assert state.resident_kv_tokens == 64
    assert state.is_genuine_decode is False


def test_worker_meta_publishes_updates_without_lazy_offload_or_kv_events():
    worker = SimpleNamespace(
        _kv_events_hook_called=False,
        get_kv_connector_kv_cache_events=lambda: None,
        lazy_offload=False,
        _pending_token_drop_updates={"td": 21},
    )

    metadata = LMCacheMPConnector.build_connector_worker_meta(worker)

    assert isinstance(metadata, LMCacheMPWorkerMetadata)
    assert metadata.resident_kv_updates == {"td": 21}
    assert worker._pending_token_drop_updates == {}
    assert LMCacheMPConnector.build_connector_worker_meta(worker) is None


def test_forward_end_collects_only_completed_compaction_reports():
    metadata = LMCacheMPConnectorMetadata()
    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(compact=lambda: {"td": 12}),
        _pending_token_drop_updates={},
        dispatcher=None,
    )
    LMCacheMPConnector.wait_for_save(worker)
    assert worker._pending_token_drop_updates == {"td": 12}
    with pytest.raises(RuntimeError, match="Duplicate token-drop"):
        LMCacheMPConnector.wait_for_save(worker)


def test_worker_does_not_publish_partial_failure():
    metadata = LMCacheMPConnectorMetadata()

    def fail():
        raise RuntimeError("compaction failed after partial write")

    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(compact=fail),
        _pending_token_drop_updates={},
        dispatcher=None,
    )
    with pytest.raises(RuntimeError, match="partial write"):
        LMCacheMPConnector.wait_for_save(worker)
    assert worker._pending_token_drop_updates == {}


def test_reset_only_meta_counts_as_nonempty():
    metadata = LMCacheMPConnectorMetadata()
    assert len(metadata) == 0
    metadata.token_drop_reset_ids.add("td")
    assert len(metadata) == 1
