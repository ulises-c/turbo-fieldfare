# Memory expectations by device, model, and context

How much unified memory to expect from a TurboFieldfare server across the two
supported model families and the full context ladder.

Generate the current table with:

```bash
python3 Scripts/memory_matrix.py                 # text
python3 Scripts/memory_matrix.py --format markdown
python3 Scripts/memory_matrix.py --validate      # check against measurements
python3 Scripts/memory_matrix.py --slots 32      # non-default expert cache
```

## What the numbers are

Resident memory is four terms:

```
resident = weights held + expert-cache slots + runtime overhead + FP16 KV
```

The KV term comes from `ArchConfig.kvFootprint`
(`Sources/TurboFieldfare/Runtime/KVCache/KVFootprintModel.swift`), the same
model the server's `--prefill off` guard and the Mac app's context menu use.
`KVFootprintModelTests` cross-checks it against `KVCacheManager`'s real
allocation at 4K/16K/64K for both families, so it tracks the allocator rather
than restating a comment.

The other three terms are calibrated against measurement, not assumed. Two
corrections matter, because both were wrong in the obvious formulation:

**The installed pack is not resident.** Gemma's pack is 13.31 GiB, but routed
experts stream through a fixed slot cache. Only `model_weights.bin`
(1,322 MiB) plus `slots x expertStride` (51 MiB at the default 16 slots) is
held. Assuming the whole pack is resident overstates Gemma by roughly 12 GiB
and wrongly reports that nothing fits on a 16 GB machine.

**KV is allocated at the context cap, not the prompt length.** Measured
2026-08-28 on the M5 Max: a server started with `--max-context 262144` and
sent a **14-token** prompt reached a **7,268 MB** footprint — within 52 MB of
the 253,952-token rung. `--max-context` is therefore the memory dial, and
sizing it for the largest prompt you might ever send costs that memory on
every request.

**Every number here is `phys_footprint`, not RSS.** On Apple Silicon the Metal
heaps backing the KV cache are mapped so that RSS does not count them. The same
Qwen server at a 256K cap reports **1,075 MB RSS** and **6,641 MB
phys_footprint** — a 6x difference. Reading `ps` output against this table will
appear to show a huge overestimate that is not real. Use
`/usr/bin/footprint -p <pid>`, or the HUD, which reports the same quantity.

## Gemma 4 26B-A4B

Marginal KV cost is 20,480 bytes per token. Only the 5 full-attention layers
scale with context; the 25 sliding-window layers are capped by the FP16 ring.

| Context | KV | Total resident | M4 16 GB | M5 Max 36 GB |
|---:|---:|---:|---|---|
| 4K | 305 MB | 2.38 GiB | fits, 8.8 GiB free | fits, 22.8 GiB free |
| 8K | 385 MB | 2.46 GiB | fits, 8.7 GiB free | fits, 22.7 GiB free |
| 16K | 545 MB | 2.62 GiB | fits, 8.6 GiB free | fits, 22.6 GiB free |
| 32K | 865 MB | 2.93 GiB | fits, 8.3 GiB free | fits, 22.3 GiB free |
| 64K | 1,505 MB | 3.55 GiB | fits, 7.6 GiB free | fits, 21.6 GiB free |
| 96K | 2,145 MB | 4.18 GiB | fits, 7.0 GiB free | fits, 21.0 GiB free |
| 128K | 2,785 MB | 4.80 GiB | fits, 6.4 GiB free | fits, 20.4 GiB free |
| 192K | 4,065 MB | 6.05 GiB | fits, 5.1 GiB free | fits, 19.1 GiB free |
| 256K | 5,345 MB | 7.30 GiB | fits, 3.9 GiB free | fits, 17.9 GiB free |

Verdicts assume 70% of unified memory is usable by one process before the
system leans on swap.

**Every rung fits on both machines, including 16 GB at full 256K.** Memory is
not what limits long context here — prefill time is. The 253,952-token rung
took about 32 minutes of prefill on the M5 Max, and the M4 is slower.

## Qwen 3.6 35B-A3B

| Context | KV | Total resident | M4 16 GB | M5 Max 36 GB |
|---:|---:|---:|---|---|
| 4K | 141 MB | 3.23 GiB | fits, 8.0 GiB free | fits, 22.0 GiB free |
| 8K | 221 MB | 3.31 GiB | fits, 7.9 GiB free | fits, 21.9 GiB free |
| 16K | 381 MB | 3.46 GiB | fits, 7.7 GiB free | fits, 21.7 GiB free |
| 32K | 701 MB | 3.78 GiB | fits, 7.4 GiB free | fits, 21.4 GiB free |
| 64K | 1,341 MB | 4.40 GiB | fits, 6.8 GiB free | fits, 20.8 GiB free |
| 96K | 1,981 MB | 5.03 GiB | fits, 6.2 GiB free | fits, 20.2 GiB free |
| 128K | 2,621 MB | 5.65 GiB | fits, 5.5 GiB free | fits, 19.5 GiB free |
| 192K | 3,901 MB | 6.90 GiB | fits, 4.3 GiB free | fits, 18.3 GiB free |
| 256K | 5,181 MB | 8.15 GiB | fits, 3.0 GiB free | fits, 17.0 GiB free |

Qwen's resident split comes from a real install measured 2026-07-31 on an
M5 24 GB host (`docs/QWEN36_PERFORMANCE.md`): 1.39 GB of mapped common
weights and 1.13 GB of routed-expert slots at 16 per layer. That measurement
independently corroborates this table's KV model to within rounding —
it reported 84 MB KV and 64 MB recurrent state at 4K, against 80 MiB
(83.9 MB) and 61 MiB (64.0 MB) predicted here.

Architectural properties that follow:

- Qwen holds a **fixed 61 MiB** of gated-DeltaNet recurrent state that does
  not grow with context, in place of 30 layers of per-token K/V rows.
- Its KV is consistently **~160 MB below Gemma's** at every rung despite being
  a larger model, because only 10 of its 40 layers keep per-token K/V.
- Its expert slots are **smaller** than Gemma's (1.13 GB vs 1.61 GB at 16
  slots), so its higher total here is driven by mapped common weights, not
  by the context.
- The FP16 ring is a **no-op** for Qwen: with no sliding-window layers,
  `--prefill off` costs it nothing. At 256K unringed it needs 5 GiB where
  Gemma needs 55 GiB. This is why the `--prefill off` guard is a KV budget
  rather than a context constant — a single context bound would reject a
  configuration Qwen can serve comfortably.

**Measured for Qwen, 2026-08-29 (this repo, M5 Max 36 GB, real
`scratch/qwen36.gturbo`):** a server at `--max-context 262144 --prefill on`,
sent a 17-token prompt, reached **6,641 MB** phys_footprint. The 256K row above
projects **8,347 MB**, so the projection is **conservative by 1,706 MB (20%)** —
the direction required, but a looser fit than Gemma's rungs (0–46 MB). The gap
is the borrowed overhead constant plus expert-cache slots that fill on demand
rather than at load. `Scripts/memory_matrix.py --validate` now fails if this
projection ever drops below the measurement.

Two Qwen facts were confirmed on the real binary at the same time: it starts at
a 256K cap **with `--prefill off`**, which Gemma is refused at identical
arguments, and its footprint is cap-driven exactly as Gemma's is.

**Still not measured for Qwen:** no retrieval or prefill-timing ladder has been
run against it, and the runtime-overhead constant in the total column remains
Gemma's. The KV column is exact and the 256K total now has one anchor; the
intermediate rungs stay projections.

## Choosing a context cap

Because allocation follows the cap, pick the smallest cap that covers your
real prompts:

- **16K (default)** — chat and short agent turns. 2.6 GiB on Gemma.
- **32K–64K** — long documents, multi-file code context. Still under 4 GiB.
- **96K+** — diagnostics and capacity work. Prefill dominates: cost grows
  superlinearly, from 4.1 ms/token at 64K to 7.6 ms/token at 256K.

## Limitations

- Totals are validated on the M5 Max only. The M4 columns are projections from
  the same model; the architecture-dependent terms (KV, weights, slots) are
  machine-independent, but the ~762 MiB runtime overhead has not been measured
  on the M4.
- Figures are for one server with one in-flight request and the default
  16-slot expert cache. Use `--slots` for other configurations.
- These are admission and residency numbers. They say nothing about retrieval
  quality at long context — see the retrieval section of
  `10-long-context.md` for what remains unmeasured there.
