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

| Context | KV |
|---:|---:|
| 4K | 141 MB |
| 8K | 221 MB |
| 16K | 381 MB |
| 32K | 701 MB |
| 64K | 1,341 MB |
| 96K | 1,981 MB |
| 128K | 2,621 MB |
| 192K | 3,901 MB |
| 256K | 5,181 MB |

Qwen's KV is exact and derived the same way, but the **total** column is
deliberately absent: the pack is not installed in this checkout, so its
resident/streamed split cannot be read from a manifest, and its runtime
overhead has never been measured. Filling those cells with Gemma's constants
would be a guess wearing a measurement's clothes. Install the pack and re-run
`--validate` to complete the table.

What is already established from the architecture:

- Qwen holds a **fixed 61 MiB** of gated-DeltaNet recurrent state that does
  not grow with context, in place of 30 layers of per-token K/V rows.
- Its KV is consistently **~160 MB below Gemma's** at every rung despite being
  a larger model, because only 10 of its 40 layers keep per-token K/V.
- The FP16 ring is a **no-op** for Qwen: with no sliding-window layers,
  `--prefill off` costs it nothing. At 256K unringed it needs 5 GiB where
  Gemma needs 55 GiB. This is why the `--prefill off` guard is a KV budget
  rather than a context constant — a single context bound would reject a
  configuration Qwen can serve comfortably.

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
