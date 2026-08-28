# Stage 2 plan: measure before porting

Status: in progress. Branch `feat/qwen36-perf`, cut from `fork-main` @ `9339539`
(Stage 1 merged via PR #3).

## The premise that turned out to be wrong

Stage 2 was planned as "cherry-pick NVMAI's individually-measured perf commits,
starting with `927c94e` because it fixes a profiler bug that invalidated every
downstream measurement." Both halves of that plan were wrong.

**`927c94e` has nothing to fix here.** It repairs an undersampling bug in an
instrument our fork does not have. The instrument is added by a *different*
commit, `f4b4c08` (env-gated per-command-buffer GPU timing). Our tree has no
`recordKernelGPU` and no per-kernel stats of any kind — only the CPU-side
`totalCb1Nanos` / `totalIoNanos` / `totalCb2Nanos` counters surfaced by
`TURBO_FIELDFARE_PHASES=1`.

**Their headline optimization is already in our tree.** `4beb74f`
("io: parallelize expert pread fills across idle CPU cores, +28% decode") adds
`DispatchQueue.concurrentPerform` to the expert fill path. We already do exactly
that, in `PreadExpertStreamer.executeExpertCachePlan`
(`Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift:281`),
reached on the decode path via `ModelExpertIO.swift:97`.

## Why this generalizes

NVMAI forked from `add22ff` (upstream #49) and upstream has moved considerably
since. Where both fixed the same bottleneck independently, NVMAI's measurement is
against a serial baseline **we no longer have**. Their percentages are therefore
not transferable — a pick can be a no-op here, or a regression, and the reported
delta says nothing either way.

Worth noting their own comment on the parallel fill path records that an earlier
`concurrentPerform` was reverted for a slot-clobbering hazard, so `4beb74f` is a
re-land rather than a clean first win.

All NVMAI figures are also M3 24 GB. On an M5 Max they are hypotheses.

## Rule adopted

Before porting any NVMAI perf commit, diff our tree for an existing equivalent.
Never carry over a reported percentage as an expected result.

## Revised order

1. **Port the instrument** (`f4b4c08` + `927c94e` together, in corrected form),
   renamed to `TURBO_FIELDFARE_KERNEL_STATS=1`, off by default with only a
   disabled branch at instrumented command-buffer completion points. It breaks
   open the `unaccounted (GPU waits)` line that
   `TURBO_FIELDFARE_PHASES=1` already prints.
2. **Measure our own decode profile** on this host.
3. **Choose optimizations from that evidence**, one lever per commit, each with a
   matched-control A/B.

## Stage 2A instrumentation contract

The opt-in is exact: only `TURBO_FIELDFARE_KERNEL_STATS=1` enables collection;
unset, empty, `0`, `false`, and other values leave it disabled. The runner
aggregates completed GPU spans by role rather than retaining one sample per
command buffer, and clears that bounded state at every reset or prompt-cache
continuation request.

The CLI writes deterministic per-role and total lines to stderr after generation
so generated stdout stays byte-identical. Roles cover embedding,
attention/router, shared expert, the optional phase-1 hit split, every routed-MoE
layer completion, and whichever fused or logits head path ran. GPU spans are
reported separately from the CPU phase footer because overlapping command
buffers make their sum unsuitable as additive wall time. Per-token values divide
by the captured fused/logits-head count, not by `newTokens`: the first generated
token is seeded by prefill and has no decode command buffer to time.

Two candidate picks flip runtime defaults — `069aed6` (expert slots 32→64 plus
pinned resident weights; our default is 16) and `7cc8b5e` (rdadvise policy on;
our default is `.off`). Per AGENTS.md these land as opt-in flags only, with any
default-flip proposal deferred until we have our own measurements.

## Reusable rig

The Gate B harness from Stage 1 is the matched-control basis: same three frozen
`real-generation-v1` prompts, same fixed seeds, two binaries staged side by side
so no rebuild sits between runs. For Stage 2 the assertion inverts — Stage 1
required byte-identical Gemma output, Stage 2 requires byte-identical output
*plus* a throughput delta attributable to exactly one lever.

## Caveat on coverage

The installed model is Gemma 4 (`scratch/gemma4.gturbo`); Qwen 3.6 is not
downloaded. Expert-I/O, slot-cache, and rdadvise work is family-agnostic and
measurable on Gemma today. Any Qwen-specific decode claim needs the ~19.6 GB
install first.
