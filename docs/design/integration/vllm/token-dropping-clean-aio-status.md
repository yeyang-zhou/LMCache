# GPU-native token dropping: clean AIO integration checkpoint

This branch is an **integration/test reference**, not a monolithic PR or a merge-ready implementation.

## Baseline

- Parent/base: `e7476a6fac22ba4ebce907b3fe3dedebb07bed06`. This contains the three frozen #5737 changes squashed into one new commit. The original #5737 branch and SHAs are unchanged.
- `upstream/dev` already includes #5736: use its `TokenDropSpec`, `parse_token_drop_spec` and entry-point plugin factory. No duplicate plugin registry, config parser or GPU KV view.
- The five refreshed features (#5741/#5742/#5743/#5568/#5663) are replayed on the clean base. APC isolation, allocator/reclaim and worker lifecycle are brought in only for the integration reference.
- Pinned vLLM runtime: **0.25.1**. Unsupported TD features fail fast: chunked prefill, speculative decoding, asynchronous scheduling, multiple GPU workers, multi-group KV.

## Runtime behavior and contracts

1. Scheduler identifies TD by the upstream request-local config parser; normal requests retain LMCache lookup/store behavior.
2. TD skips LMCache lookup/offload, and must not read or publish shared vLLM APC blocks; private GPU blocks are required.
3. Scheduler owns authoritative physical resident frontier **P**, separate from logical progress **L** and RoPE/model positions. Allocator accounts for P; worker adaptor uses P only for KV read/write positions and lengths.
4. Request metadata carries algorithm selection, current P, scheduler-owned block IDs, decode facts and a mandatory scheduler-validated exclusive/private block gate.
5. Worker initializes per-request algorithm plugins, optionally captures post-RoPE Q, and selects ordered kept positions in `[kept_tokens, kv_heads]` layout. Reuse frozen `PagedKVSequenceView`; no extra contiguous K/V requirement introduced by LMCache.
6. Validate all layer/head selection shapes and index ranges before the first destructive write. Selection errors can skip before mutation if the CUDA context is healthy. Compaction errors after possible in-place writes must propagate, **not** silently resume decoding.
7. Worker synchronizes its compaction stream before publishing absolute resident-length updates; scheduler commits P and frees only now-unneeded private tail blocks after receiving the worker output.
8. Clean algorithm/request state on finish and preemption; regular requests remain vanilla in mixed batches.

## Maintainer-review constraints

- Kuntai: CUDA-only tests for GPU operations (not cross-device coverage in this MVP).
- Kuntai: no forced `.contiguous()` in the frozen KV accessor; downstream algorithms decide when materialization is needed.
- Kuntai: serving availability matters, but catching exceptions after possible in-place mutation is not a safe fallback. Fall back only before mutation and with a healthy CUDA context; fail fast otherwise.
- Kuntai + ApostaC: code placement under `integration/vllm/experimental` is accepted.
- Merged #5736 already implements the request-local algorithm factory and parser; do not reinvent them.

## Validation gates / release blockers

- **Static**: Ruff lint + formatting, isort, compileall, Git diff whitespace.
- **CUDA unit**: both FP16/BF16; invalid indices prewrite; mixed TD+normal; repeated compactions; independent per-head/layer kept positions; failed GPU operations never publish P. GPU-required tests use CUDA-only skip markers.
- **Pinned vLLM runtime**: import and smoke-test adaptor signatures, worker hook timing, attention Q capture, cache layout, metadata path and scheduling; verify request finish/preemption/ID reuse.
- **End-to-end correctness**: compare normal-only, TD-only, mixed and alternating batches; non-dropped KV matches vanilla, drop moves K/V correctly, next decode uses P, tail reclaimed only after worker barrier; no APC or LMCache offload leakage.
- **Performance**: benchmark vanilla versus mixed TD, GPU compaction microbenchmark (including paged materialization, scoring and relocation), and throughput/latency at matched workload/buffer settings.
- **R-KV plugin compatibility**: check the latest Zefan-Cai/R-KV serving code and pending token-major PR #64; do not assume this branch alone proves plugin API compatibility.
- The connected Windows worktree does not have CUDA, `torch`, `vllm` or `pytest`. Any CUDA/E2E/perf gates not executed there remain explicitly unverified.

This is the reference to refine with the next R-KV serving changes **before** finalizing the BFS-style PR dependency split.
