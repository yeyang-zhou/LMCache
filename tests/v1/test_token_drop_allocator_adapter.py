# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _allocate_with_resident_frontier,
    install_token_drop_allocator_adapter,
)
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


def _original_allocate(
    manager,
    request,
    num_new_tokens,
    num_new_computed_tokens=0,
    new_computed_blocks=None,
    num_lookahead_tokens=0,
    num_external_computed_tokens=0,
    delay_cache_blocks=False,
    num_encoder_tokens=0,
    full_sequence_must_fit=False,
    reserved_blocks=0,
    has_scheduled_reqs=True,
):
    manager.seen_request = request
    manager.seen_num_computed_tokens = request.num_computed_tokens
    manager.seen_num_new_tokens = num_new_tokens
    manager.seen_delay_cache_blocks = delay_cache_blocks
    return manager.result


def _request(logical: int, *, token_drop: bool = True):
    transfer = (
        {
            "lmcache.token_drop": {
                "algorithm": "fake",
                "config": {},
            }
        }
        if token_drop
        else {}
    )
    return SimpleNamespace(
        request_id="req",
        num_computed_tokens=logical,
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": transfer}),
    )


@pytest.fixture(autouse=True)
def _clear_resident_state():
    clear_resident_kv_tokens("req")
    yield
    clear_resident_kv_tokens("req")


def test_resident_frontier_is_used_only_for_token_drop_allocation():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    result = _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        1,
    )

    assert result is manager.result
    assert manager.seen_num_computed_tokens == 32
    assert manager.seen_request is not request
    assert manager.seen_request.physical_kv_length == 32
    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 33


def test_normal_request_is_exact_vanilla_passthrough():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104, token_drop=False)
    set_resident_kv_tokens(request.request_id, 32)

    result = _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        1,
        delay_cache_blocks=True,
    )

    assert result is manager.result
    assert manager.seen_num_computed_tokens == 104
    assert manager.seen_request is request
    assert manager.seen_delay_cache_blocks is True
    assert get_resident_kv_tokens(request.request_id) is None


def test_failed_allocation_does_not_advance_resident_frontier():
    manager = SimpleNamespace(max_model_len=4096, result=None)
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    assert (
        _allocate_with_resident_frontier(
            _original_allocate,
            manager,
            request,
            1,
        )
        is None
    )

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 32


def test_preemption_returns_to_vanilla_frontier_without_changing_cache_flag():
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(0)
    set_resident_kv_tokens(request.request_id, 32)

    _allocate_with_resident_frontier(
        _original_allocate,
        manager,
        request,
        8,
        delay_cache_blocks=False,
    )

    assert manager.seen_num_computed_tokens == 0
    assert manager.seen_delay_cache_blocks is False
    assert get_resident_kv_tokens(request.request_id) is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_new_computed_tokens", 1),
        ("new_computed_blocks", object()),
        ("num_lookahead_tokens", 1),
        ("num_external_computed_tokens", 1),
        ("num_encoder_tokens", 1),
        ("full_sequence_must_fit", True),
    ],
)
def test_resident_frontier_rejects_unsupported_allocation_modes(name, value):
    manager = SimpleNamespace(max_model_len=4096, result=object())
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    with pytest.raises(ValueError, match="normal synchronous full-attention"):
        _allocate_with_resident_frontier(
            _original_allocate,
            manager,
            request,
            1,
            **{name: value},
        )

    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 32


def test_install_allocator_adaptor_is_idempotent_for_pinned_vllm():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__ != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_allocate = KVCacheManager.allocate_slots
    try:
        install_token_drop_allocator_adapter()
        wrapped_allocate = KVCacheManager.allocate_slots

        assert getattr(
            wrapped_allocate,
            "_lmcache_token_drop_allocator_adapter",
            False,
        )
        install_token_drop_allocator_adapter()
        assert KVCacheManager.allocate_slots is wrapped_allocate
    finally:
        KVCacheManager.allocate_slots = original_allocate


def test_resident_state_helpers():
    assert get_resident_kv_tokens("req") is None
    set_resident_kv_tokens("req", 7)
    assert get_resident_kv_tokens("req") == 7
    clear_resident_kv_tokens("req")
    assert get_resident_kv_tokens("req") is None


def test_physical_allocation_exception_keeps_logical_request_unchanged():
    manager = SimpleNamespace(max_model_len=4096)
    request = _request(104)
    set_resident_kv_tokens(request.request_id, 32)

    def fail(_manager, allocation_request, *_args):
        assert allocation_request is not request
        assert allocation_request.num_computed_tokens == 32
        assert request.num_computed_tokens == 104
        raise RuntimeError("injected allocation failure")

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        _allocate_with_resident_frontier(fail, manager, request, 1)
    assert request.num_computed_tokens == 104
    assert get_resident_kv_tokens(request.request_id) == 32
