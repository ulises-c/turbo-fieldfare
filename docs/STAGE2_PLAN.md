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
   `TURBO_FIELDFARE_PHASES=1` already prints. — **done**, `a73c25b` + `f2a8e6c`.
2. **Measure our own decode profile** on this host. — **done**, see Stage 2B
   below. Expert I/O await is the flat ~40 % block; attention over KV is the
   context-dependent one.
3. **Choose optimizations from that evidence**, one lever per commit, each with a
   matched-control A/B. — in progress. Expert-cache slots and rdadvise A/B'd on
   both families (Gemma Stage 2B, Qwen Stage 2C): slots is a real opt-in win
   (+4.3 % Gemma, +26 % Qwen), rdadvise is a settled negative. No default flipped.

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

## Stage 2B: our own decode profile

M5 Max MacBook Pro, 36 GB, macOS 26.6.2, Swift 6.3.3, commit `f2a8e6c`, release
build, Gemma 4 `scratch/gemma4.gturbo`. All three frozen `real-generation-v1`
prompts, `--max-new 128 --max-context 4096 --temperature 0.2 --top-k 64
--top-p 0.95 --seed 20260721`, defaults otherwise. Instrumentation on via
`TURBO_FIELDFARE_KERNEL_STATS=1 TURBO_FIELDFARE_PHASES=1`.

| prompt | prefill | decode | tok/s | expert io await | unaccounted | kernel total |
|---|---|---|---|---|---|---|
| short-explanation | 61tok/5.15s | 3.02s | 42.44 | 1222.8 ms | 1692.5 ms | 1382.1 ms |
| medium-review | 430tok/6.31s | 3.18s | 40.30 | 1198.1 ms | 1872.5 ms | 1608.1 ms |
| long-synthesis | 3015tok/16.12s | 3.35s | 38.17 | 1192.2 ms | 2059.6 ms | 1740.0 ms |

Three repeats of short-explanation landed at 42.44 / 43.70 / 43.65 tok/s with
byte-identical stdout, so run-to-run noise is well under the effects below.

Reading it:

1. **Expert I/O await is ~40 % of decode and flat across prompt lengths**
   (1222/1198/1192 ms). It does not scale with context, so it is a per-token
   fixed cost of pulling routed experts, not a context effect. This is the
   largest single addressable block.
2. **`unaccounted (GPU waits)` is the other ~55 %, and it is the part that
   grows with context** (1692 → 2059 ms). The kernel report explains why:
   `attention_router` alone goes 527 → 684 → 903 ms across the same prompts
   while every other role stays flat. Attention over a longer KV is the
   context-dependent term; the MoE roles are not.
3. **Summed GPU spans (1382 ms) are well under decode wall time (3016 ms)** even
   though the roles cover the whole forward pass — command buffers overlap, so
   the gap is stall, not unmeasured compute. Consistent with I/O await being
   real waiting rather than hidden GPU work.
4. `embed` is 0.36 ms total over 128 tokens. Nothing to win there.

So the two levers worth an A/B are the two blocks above: expert-cache residency
(attacks block 1) and read-advice (also block 1, via the same await).

## Stage 2B A/B: expert-cache slots

First pass ran the arms back to back and every run beat the previous one
regardless of arm — page-cache warming aliased perfectly onto run order. Redone
with two warmup runs discarded and the arms interleaved A/B/A/B so order cannot
alias the lever. Four pairs, short-explanation, all ten stdouts share one hash.

| arm | tok/s (4 runs) | mean | expert io await | unaccounted |
|---|---|---|---|---|
| `--expert-cache-slots 16` (default) | 45.93 / 44.90 / 45.55 / 45.43 | **45.45** | 1078.7 ms | 1644.9 ms |
| `--expert-cache-slots 32` | 48.20 / 46.80 / 46.38 / 48.17 | **47.39** | 866.0 ms | 1753.2 ms |

**+1.93 tok/s, +4.3 %.** 32 slots wins all four adjacent pairs and the two
distributions are disjoint (min B 46.38 > max A 45.93), so the effect clears the
noise floor.

The mechanism matches the profile rather than merely correlating with it: the
gain is bought entirely in expert I/O await, −212.7 ms (−19.7 %), exactly the
block the profile named. Note the partial refund — `unaccounted` rises +108.4 ms
(+6.6 %) as decode stalls less on I/O and more on the GPU, so about half the I/O
saving is given back and the headline is +4.3 % rather than the ~7 % the I/O
delta alone would suggest.

Read-advice (`--rdadvise adaptive` / `bounded`) cut I/O await similarly but
returned it all to `unaccounted`, netting no throughput change. It was measured
only in the order-confounded first pass, so it is recorded as "no effect
observed", not as a settled negative. **(Update: re-measured cleanly on Qwen in
Stage 2C below — now a settled negative, −10.4 % tok/s for `adaptive`.)**

Per AGENTS.md this is a measurement, not a default change: 16 slots remains the
default and `--expert-cache-slots 32` remains opt-in. A default-flip proposal
needs the other two prompts, a memory-headroom check at 32 slots, and the same
interleaved treatment before it is worth writing.

## Stage 2C: the Qwen 3.6 family

Qwen 3.6 35B-A3B is now installed (`scratch/qwen36.gturbo`), so the family caveat
below is closed. The instrument, the profile, and the slot A/B were all reproduced
on Qwen with the same rig and treatment as Gemma. Scripts:
`Scripts/stage2_qwen_profile.sh`, `Scripts/stage2_qwen_ab.sh`,
`Scripts/stage2_qwen_rss.sh`. Raw numbers in
`docs/experiments/data/stage2c-qwen36-decode.jsonl`.

### Baseline decode profile

M5 Max, 36 GB, release build, `--max-new 128 --max-context 4096 --temperature 0.2
--top-k 64 --top-p 0.95`, one warmup discarded per case, instrumentation on.

| prompt | prefill | decode | tok/s | expert io await | unaccounted | attn_router | kernel total |
|---|---|---|---|---|---|---|---|
| short-explanation | 62tok/21.75s | 9.02s | 14.19 | 3130.4 ms | 5408.2 ms | 2665.1 ms | 4617.7 ms |
| medium-review | 426tok/24.87s | 8.54s | 14.99 | 2923.1 ms | 5142.5 ms | 2527.5 ms | 4306.7 ms |
| long-synthesis | 2940tok/50.30s | 8.73s | 14.67 | 3239.6 ms | 5003.4 ms | 2566.0 ms | 4068.2 ms |

Qwen decodes at ~14–15 tok/s here, roughly a third of Gemma's ~40, and the shape
differs: **expert I/O await is a much larger block (2.9–3.2 s vs Gemma's ~1.2 s)**
and, like Gemma, it is flat across prompt length — a per-token fixed cost of
pulling routed experts, not a context effect. `attention_router` is also flat
(2.5–2.7 ms-scale), but note these runs cap context at 4096, so the KV-growth term
Gemma showed is not exercised here. The larger await block is the reason to expect
the slot lever to matter *more* on Qwen than on Gemma.

### Stage 2C A/B: expert-cache slots (Qwen)

Same treatment as the Gemma A/B: two warmups discarded, arms interleaved
A/B/A/B, four pairs, short-explanation. All eight measured stdouts share one hash
(`9edfd32d64764e62`), so output is byte-identical across the lever.

| arm | tok/s (4 runs) | mean | expert io await | unaccounted | peak RSS |
|---|---|---|---|---|---|
| `--expert-cache-slots 16` (default) | 13.78 / 13.82 / 14.18 / 15.61 | **14.35** | 3231.6 ms | 5235.6 ms | 1.37 GiB |
| `--expert-cache-slots 32` | 16.67 / 17.46 / 19.79 / 18.37 | **18.07** | 2366.9 ms | 4316.7 ms | 2.44 GiB |

**+3.73 tok/s, +26.0 %.** 32 slots wins all four adjacent pairs and the
distributions are disjoint (min B 16.67 > max A 15.61), so the effect clears the
noise floor decisively. Both arms drift upward run-over-run (the machine keeps
warming past the two warmups), but because the arms are interleaved that drift
cancels in the pairwise comparison — it does not manufacture the gap.

Two ways this differs from Gemma, both material:

1. **The win is ~6× larger** (+26 % vs +4.3 %), because expert I/O await is a far
   bigger share of Qwen's decode, exactly as the profile predicted.
2. **There is no refund.** On Gemma about half the I/O saving reappeared as
   `unaccounted` GPU stall. On Qwen *both* blocks drop — await −864.7 ms (−26.8 %)
   **and** unaccounted −918.9 ms (−17.6 %) — so the throughput gain tracks the full
   I/O saving rather than half of it. This is why measuring per-family was worth
   doing: the Gemma result would have understated Qwen by a wide margin.

### Still opt-in, and why

Per AGENTS.md this is a measurement, not a default change. The cost side is real:
32 slots raises peak RSS from 1.37 GiB to 2.44 GiB, **+1.07 GiB (+78 %)**, for this
model at 4K context. On a memory-constrained device that is a genuine tradeoff, so
16 slots stays the default and `--expert-cache-slots 32` stays opt-in. A defensible
default-flip proposal would still need the other two Qwen prompts under the same
interleaved treatment and a headroom check at the largest supported context, not
just 4K.

### Stage 2C A/B: read-advice (rdadvise), the un-confounding

The Gemma pass left rdadvise as "no effect observed" because its only measurement
was order-confounded. Re-run cleanly on Qwen with the slot-A/B treatment (two
warmups discarded, `off` vs `adaptive` interleaved A/B, four pairs,
short-explanation), it is now a **settled negative**, not an unknown. Script:
`Scripts/stage2_qwen_rdadvise.sh`. All eight stdouts share the same hash.

| arm | tok/s (4 runs) | mean | expert io await | unaccounted |
|---|---|---|---|---|
| `--rdadvise off` (default) | 14.59 / 14.82 / 14.15 / 14.72 | **14.57** | 3234.3 ms | 5071.9 ms |
| `--rdadvise adaptive` | 13.31 / 13.07 / 12.81 / 13.05 | **13.06** | 2821.4 ms | 5991.6 ms |

**−1.51 tok/s, −10.4 %** — adaptive is consistently *worse*, losing all four
adjacent pairs with disjoint distributions. The mechanism is now measured rather
than guessed: adaptive read-advice **does** cut expert I/O await (−412.9 ms,
−12.8 %), but that saving is **more than fully refunded to `unaccounted`**
(+919.7 ms, +18.1 %). The madvise hints trade I/O wait for GPU stall and then
some, so throughput drops. `off` is correctly the default and there is no case
here for changing it. (Only `adaptive` was tested; `bounded`/`default` were not,
but the await-for-stall refund makes a net win from a gentler policy unlikely.)

## Reusable rig

The Gate B harness from Stage 1 is the matched-control basis: same three frozen
`real-generation-v1` prompts, same fixed seeds, two binaries staged side by side
so no rebuild sits between runs. For Stage 2 the assertion inverts — Stage 1
required byte-identical Gemma output, Stage 2 requires byte-identical output
*plus* a throughput delta attributable to exactly one lever.

## Caveat on coverage

Both families are now installed (`scratch/gemma4.gturbo`, `scratch/qwen36.gturbo`).
Expert-I/O, slot-cache, and rdadvise work is family-agnostic and has now been
measured on **both** — see Stage 2C above for the Qwen profile and slot A/B. The
remaining gaps are per-family breadth (only short-explanation was A/B'd on Qwen;
medium and long still want the same interleaved treatment), not a missing model
or an unmeasured rdadvise.
