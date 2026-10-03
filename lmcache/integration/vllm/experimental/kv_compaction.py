# SPDX-License-Identifier: Apache-2.0
"""Apply in-place paged-KV compaction copies."""

from __future__ import annotations

import torch

_MAX_COMPACTION_PAYLOAD_BYTES = 32 * 1024 * 1024


def apply_kv_compaction_moves(
    kv_cache: torch.Tensor,
    physical_slot_copies: torch.Tensor,
) -> None:
    """Copy paged-KV entries in place.

    kv_cache is shaped [num_blocks, slots_per_block, ...].
    physical_slot_copies is [num_copies, 2] source/destination physical-slot
    pairs in compaction-safe order. Trailing dimensions move together as one
    KV entry.
    """
    if physical_slot_copies.numel() == 0:
        return

    slots_per_block = kv_cache.shape[1]
    entry_bytes = kv_cache[0, 0].numel() * kv_cache.element_size()
    copies_per_chunk = max(1, _MAX_COMPACTION_PAYLOAD_BYTES // entry_bytes)

    for start in range(0, physical_slot_copies.shape[0], copies_per_chunk):
        chunk = physical_slot_copies[start : start + copies_per_chunk]
        source_slots, destination_slots = chunk.unbind(dim=1)

        # Gather before writing so overlapping copies inside one chunk read
        # their original source values.
        payload = kv_cache[
            source_slots // slots_per_block,
            source_slots % slots_per_block,
        ]
        kv_cache[
            destination_slots // slots_per_block,
            destination_slots % slots_per_block,
        ] = payload
        del payload
