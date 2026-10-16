# GPU-native token dropping: refreshed AIO status

Snapshot: 2026-10-16. **Integration reference; not a mergeable monolithic PR.**

## Provenance

- Base: LMCache `dev` at `e64f0a56`, already containing the config/plugin factory merged through #5736.
- Frozen GPU primitive: #5737 at `133fe3c9`, including its exact three original commits. Do **not** modify or reimplement it.
- Refreshed integration ancestry: #5741 (request-local selection and lookup bypass), #5742 (STORE/offload bypass and preserved lazy-offload registry lifecycle), #5743 (absolute resident-length metadata), #5568 (worker request state), and #5663 (physical KV addressing).
- Earlier discussion-only AIO #5563 is **source material only**, not a branch to merge or blindly replay.

## Added in this refreshed AIO worktree

- Version-pinned vLLM resident-frontier allocator adapter; logical token position remains unchanged.
- Request-selective APC lookup/publication guard. No global APC disable.
- Request-local worker policy, post-RoPE Q observations, and per-layer KV compaction.
- **One** GPU KV primitive: `PagedKVSequenceView` from frozen #5737. Getters are token-major `[tokens, kv_heads, head_dim]`; retained indices use `[kept_tokens, kv_heads]`. No legacy AIO `KVView`/copy engine.
- Worker validates all layer/head indices, retained counts and current-resident bounds before writes. Pre-write algorithm errors can skip compaction only when CUDA synchronization succeeds. After writes, GPU completion must succeed before resident length is returned; failure propagates instead of catch-and-continue.

## Not yet wired or validated

These modules are **not** automatically activated by this branch. Remaining orchestration must enforce all gates before claiming runtime readiness:

1. Scheduler admission and per-request option validation for supported pinned vLLM modes.
2. APC guard + allocator installation, exclusive/unhashed block ownership at the destructive mutation boundary, and correct request/worker block IDs.
3. Forward Q-observation lifecycle and worker compaction hookup; request finish, preempt and reused-ID cleanup.
4. Worker-output publication and scheduler commit of **absolute** resident length only after the GPU completion barrier; reclaim private tail blocks.
5. Recoverable pre-mutation fallback versus request abort when safe isolation is provable. Never treat possibly partial KV writes as a no-op; engine-level failure remains necessary where request isolation is not supported.
6. H100 CUDA BF16/FP16 and pinned-vLLM 0.25.1 E2E tests: mixed normal+TD, repeated compaction, APC/preempt/reuse, logical vs physical addressing, failure injection, and parity.

**Testing statement:** static Ruff, isort, format, and Python compile checks are available from the preparation environment. That Windows environment does not have torch/vLLM/CUDA, so neither the new GPU tests nor serving E2E can be claimed to pass. Frozen #5737 testing remains as recorded in its own PR.

## Collaboration

This branch is a refreshed integration sandbox, not a replacement for atomic Draft PRs. #5737 stays frozen, and the original #5563 AIO branch stays untouched. Any runtime activation or extraction into production PRs requires independent review and the correctness gates above.
