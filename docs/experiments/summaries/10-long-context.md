# Long-context ladder and retrieval

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Server guide](../../OPENAI_SERVER.md) |
[Runtime controls](../../RUNTIME_CONTROLS.md)

Extending the server's `--max-context` allowlist to the Gemma 4 26B-A4B native
256K raised two separate questions. Whether a prompt that long is *admitted* —
accepted, prefilled, and memory stable — and whether it is *used*, meaning the
model still attends across it. They need different harnesses and they are
reported separately here.

| Current result | Disposition |
| --- | --- |
| 256K admission, prefill, and memory stability | Measured on M5 Max; see caveats below |
| `--prefill off` above 64K | Rejected during argument parsing |
| Long-context retrieval at ladder rungs | **15/15 recall, 57K–253K, all depths** |
| Prefill timing on the current build | Re-measured, matched, one binary |
| Multi-token decode throughput at long context | Not measured; probes use a short completion |

## Why the sliding window makes this two questions

`ArchConfig.gemma4_26B_A4B` sets `slidingWindow: 1024` and marks layers 5, 11,
17, 23, and 29 as full attention. Twenty-five of thirty layers therefore see a
1024-token window, and every long-range dependency rests on five layers.

Raising the context cap allocates KV and admits the prompt. It does not by
itself widen what those five layers can carry. A model whose long-range
attention had degraded would still return HTTP 200 with a well-formed
one-token completion, so the ladder below cannot detect that failure and must
not be read as evidence against it.

## Admission ladder

`Scripts/context_ladder.py` starts a fresh release server per rung, submits one
request with `max_completion_tokens=1`, and samples memory every ten seconds. A
rung passes on HTTP 200 with the intended prompt token count, no rejection,
timeout, cancellation, or OOM. Resource pressure is recorded rather than used
as a hidden threshold.

The prompt is `"word " * N`. That makes the token count reproducible, and it is
also the measurement's main limitation: repeated filler is not representative
of agent history and can produce unusually stable expert routing.

### Recorded run

Apple M5 Max MacBook Pro, 18 CPU cores, 32-core GPU, 36 GB unified memory,
macOS 26.5.2 (25F84), Apple Swift 6.3.3. Reserve was a uniform 8,192 tokens
across every rung.

| Context cap | Actual prompt | E2E server time | Prefill | Prefill tok/s | Peak footprint | Peak Metal | Minimum free memory | Swap | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64K | 57,344 | 236.158s | 236.000s | 242.983 | 3,441 MB | 1,537 MB | 76% | 0 MB | completed |
| 96K | 90,112 | 464.218s | 463.937s | 194.233 | 4,082 MB | 2,177 MB | 76% | 0 MB | completed |
| 128K | 122,880 | 701.019s | 700.724s | 175.361 | 4,714 MB | 2,817 MB | 75% | 0 MB | completed |
| 192K | 188,416 | 1,291.620s | 1,290.896s | 145.958 | 6,032 MB | 4,097 MB | 68% | 0 MB | completed |
| 256K | 253,952 | 1,920.565s | 1,919.868s | 132.276 | 7,320 MB | 5,377 MB | 62% | 0 MB | completed |

Prompt length grows 4.43x from 64K to 256K while end-to-end time grows 8.13x.
Time per prompt token rises from 4.118 ms to 7.563 ms and prefill throughput
falls from 242.983 to 132.276 tokens/s, so cost is superlinear in length even
though a least-squares fit over these five points is close to linear
(R² = 0.996). No rung touched swap.

### Provenance and a known staleness

These numbers were produced before this branch merged `main`, which brought in
"Speed up prefill on pre-Apple10 Macs" (#159). That change rewrote prefill
attention pipeline selection and the prefill command-buffer structure, which is
exactly the path the timing columns measure.

**The timing columns above therefore describe a superseded prefill
implementation.** They are retained as a dated lab record only. The memory
columns are structural — they follow from KV allocation, not from scheduling —
and are expected to hold; see the check below. The rungs were also run across
more than one commit, with the 64K baseline recorded last, so they are not a
matched set even among themselves.

**Superseded.** "Prefill timing, re-measured" below carries a matched
single-binary replacement, three probes per rung. Quote that table, not this
one.

### Memory model cross-check

FP16 KV allocation follows directly from `KVCacheManager`: with the ring
enabled, each of the 25 sliding-window layers holds
`slidingWindow + prefillChunkTokens` = 1,152 tokens at
`numKVHeads * headDim * 2` bytes, while each of the 5 full-attention layers
holds `maxContext` at `numFullKVHeads * fullHeadDim * 2` bytes, doubled for K
and V:

```
KV(ctx) = 25 * 1152 * 4096 * 2  +  5 * ctx * 2048 * 2
```

| Context | Predicted KV | Observed peak Metal | Difference |
|---:|---:|---:|---:|
| 64K | 1,505 MB | 1,537 MB | +32 MB |
| 96K | 2,145 MB | 2,177 MB | +32 MB |
| 128K | 2,785 MB | 2,817 MB | +32 MB |
| 192K | 4,065 MB | 4,097 MB | +32 MB |
| 256K | 5,345 MB | 5,377 MB | +32 MB |

A constant 32 MB offset across a 4.43x span means the model accounts for the
scaling term and the residual is fixed overhead. This is also why
`--prefill off` is rejected above 64K: without the ring every layer allocates
at `maxContext`, and the same formula gives about 55 GiB at 256K.

## Retrieval

`Scripts/context_retrieval.py` answers the second question. It plants a
distinctive fact at a fractional depth in filler, asks for it back with a real
multi-token completion, and sweeps depth because sliding-window degradation is
position dependent — a needle near the end stays inside every layer's local
window, while one in the middle must survive on the full-attention layers.

### Recorded run

Same machine as above, single binary at commit `b45ee76`.

| Context cap | Actual prompt | Depths | Recall |
|---:|---:|---|---|
| 4K | 1,543 | 0.1 / 0.5 / 0.9 | 3/3 |
| 16K | 14,043 | 0.1 / 0.5 / 0.9 | 3/3 |
| 64K | 57,043 | 0.1 / 0.5 / 0.9 | 3/3 |
| 96K | 89,843 | 0.1 / 0.5 / 0.9 | 3/3 |
| 128K | 122,543 | 0.1 / 0.5 / 0.9 | 3/3 |
| 192K | 187,843 | 0.1 / 0.5 / 0.9 | 3/3 |
| 256K | 253,143 | 0.1 / 0.5 / 0.9 | 3/3 |

**15/15 across the ladder, no misses.** Depth 0.5 is the load-bearing case:
at 253,143 tokens the needle sits roughly 126,000 tokens from either end, far
outside the 1,024-token window of all 25 sliding-window layers, so it is
reachable only through the 5 full-attention layers. It was recovered verbatim
at every rung. Extending the cap therefore extends what the model can
actually retrieve, not merely what it will accept.

Every probe reported `cached_tokens: 0`, so no prefix cache shortened a
prefill and each result is an independent measurement.

**Scope.** Filler is repetitive by construction, which is the easy case for
retrieval: a single distinctive sentence in a field of identical words. Real
agent history is far more confusable. Read this as an upper bound on
long-context recall, and as strong evidence against catastrophic
sliding-window failure — not as a guarantee for arbitrary content.

## Prefill timing, re-measured

The timing table earlier in this document predates the merge of #159 and was
recorded across several commits. The retrieval sweep re-measures it as a
by-product: every probe records `pp_seconds` and `pp_tokens_per_second`, all
on one binary in one sitting, three probes per rung.

| Context cap | Prompt | Pre-#159 tok/s | Measured tok/s (median of 3) | Delta |
|---:|---:|---:|---:|---:|
| 64K | 57,043 | 242.98 | 214.78 | −11.6% |
| 96K | 89,843 | 194.23 | 185.25 | −4.6% |
| 128K | 122,543 | 175.36 | 163.94 | −6.5% |
| 192K | 187,843 | 145.96 | 136.89 | −6.2% |
| 256K | 253,143 | 132.28 | 118.88 | −10.1% |

Median delta −6.5%, mean −7.8%, same sign at every rung. Medians are quoted
because 128K depth 0.5 returned a single 128.2 tok/s outlier against 165.6 and
163.9 at the other two depths; the other four rungs vary by 1.6 to 3.2 tok/s
across depths, so that one point is a transient rather than depth sensitivity.

**This is not a like-for-like regression measurement.** The comparison is
against numbers recorded on a different, superseded build across several
commits, with the 64K baseline taken last — so the old column is not an
internally matched set either. #159 targeted pre-Apple10 Macs, and this host
is an M5 Max, so no speedup was expected here; whether these few percent are
the cost of that fix, run-to-run variance, or thermal state is unresolved.
Settling it needs an A/B against the pre-#159 commit on this machine, which
has not been run.

What the new column *is* good for: it is a matched, single-binary,
three-probe-per-rung baseline. Future changes should be compared against it,
not against the pre-#159 numbers.

The shape of the curve is unchanged and is the practically important part.
Time per prompt token rises from 4.60 ms at 64K to 8.46 ms at 256K, so a full
256K prefill costs about 35 minutes on this host. **Memory is not the limit at
long context on either of the tested devices; prefill time is.**

## Reproducing

```bash
swift build -c release --product TurboFieldfareServer

# Admission ladder: all rungs, or one at a time merged into the aggregate.
python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo
python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo --levels 256

# Retrieval at a given cap.
python3 Scripts/context_retrieval.py --model scratch/gemma4.gturbo \
  --max-context 262144 --filler-words 250000 --depths 0.1,0.5,0.9

# Recall AND matched prefill timing across the whole ladder, one binary,
# one server per rung. This is what produced the two tables above.
python3 Scripts/context_sweep.py
python3 Scripts/context_sweep.py --rungs 65536,98304
```

Both harnesses write JSON under `benchmark-results/`, which is ignored by git;
copy any result worth keeping into this note. Every rung is only comparable to
rungs produced by the same binary, so record the commit and re-measure the
whole ladder after any change to the prefill or attention path.
