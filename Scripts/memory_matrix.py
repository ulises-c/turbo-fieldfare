#!/usr/bin/env python3
"""Project resident memory across device, model, and context length.

The KV column is not a curve fit. It comes from the same analytic model the
runtime uses (`ArchConfig.kvFootprint` in
Sources/TurboFieldfare/Runtime/KVCache/KVFootprintModel.swift), which
`KVFootprintModelTests` cross-checks against `KVCacheManager`'s real
allocation at 4K/16K/64K for both families. That check is what makes this a
prediction rather than a guess.

Everything outside KV is calibrated against the five measured Gemma ladder
rungs (docs/experiments/summaries/10-long-context.md), not assumed:

  * The full weight file is NOT resident. Routed experts stream through a
    fixed slot cache, so only `model_weights.bin` plus `slots * expertStride`
    is held. For Gemma that is 1,322 MiB + 51 MiB at the default 16 slots,
    against a 13.3 GiB installed size.
  * A further ~550 MiB of runtime overhead (activations, scratch, logits
    head, tokenizer) is constant across the rungs: the measured non-KV
    residency sits at 1,897-1,943 MiB over a 4.4x span in context.

Qwen 3.6 KV is exact, but its non-KV residency is UNMEASURED: the pack is not
installed in this checkout, so its resident/streamed split cannot be read from
a manifest. Those cells are reported as unknown rather than filled with a
Gemma-shaped guess.

Usage:
    python3 Scripts/memory_matrix.py
    python3 Scripts/memory_matrix.py --format markdown
    python3 Scripts/memory_matrix.py --slots 32
"""

import argparse

FP16 = 2
FP32 = 4
KEY_AND_VALUE = 2
GIB = 1024 ** 3
MIB = 1024 ** 2

# Measured non-KV runtime overhead beyond resident weights and expert slots.
# Calibrated so the projection never under-predicts a measured rung: the
# observed non-KV residency across the five Gemma rungs was 1,897-1,943 MiB,
# of which 1,373 MiB is resident weights + 16 expert slots, leaving 524-570
# MiB. Using the 570 MiB midpoint still under-predicted every rung by
# 146-192 MB, so the bound is set at the observed maximum plus that gap.
# Validate with --validate after changing this.
RUNTIME_OVERHEAD_BYTES = 762 * MIB


class Arch:
    def __init__(self, name, mask, num_kv_heads, head_dim, num_full_kv_heads,
                 full_head_dim, sliding_window, resident_weight_bytes,
                 expert_stride, installed_bytes, linear=None, measured=True,
                 overhead_is_borrowed=False):
        self.name = name
        self.mask = mask
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_full_kv_heads = num_full_kv_heads
        self.full_head_dim = full_head_dim
        self.sliding_window = sliding_window
        self.resident_weight_bytes = resident_weight_bytes
        self.expert_stride = expert_stride
        self.installed_bytes = installed_bytes
        self.linear = linear
        # False when the resident/streamed split has not been measured.
        self.measured = measured
        # True when the runtime-overhead constant was measured on a DIFFERENT
        # architecture and carried over, so totals are projections.
        self.overhead_is_borrowed = overhead_is_borrowed

    @property
    def sliding_bytes_per_token(self):
        return self.num_kv_heads * self.head_dim * KEY_AND_VALUE * FP16

    @property
    def full_bytes_per_token(self):
        return self.num_full_kv_heads * self.full_head_dim * KEY_AND_VALUE * FP16

    @property
    def linear_state_bytes(self):
        if not self.linear:
            return 0
        n = sum(1 for m in self.mask if m == 2)
        state = (self.linear["num_v_heads"] * self.linear["value_head_dim"]
                 * self.linear["key_head_dim"] * FP32)
        qkv_dim = (2 * self.linear["num_k_heads"] * self.linear["key_head_dim"]
                   + self.linear["num_v_heads"] * self.linear["value_head_dim"])
        conv_tail = max(0, self.linear["conv_kernel_size"] - 1) * qkv_dim * FP16
        return n * (state + conv_tail)

    @property
    def marginal_kv_bytes_per_token(self):
        return sum(1 for m in self.mask if m == 1) * self.full_bytes_per_token

    def kv_bytes(self, ctx, chunk_tokens=128, ring=True):
        rows = min(ctx, max(1, self.sliding_window + chunk_tokens)) if ring else ctx
        total = 0
        for m in self.mask:
            if m == 2:
                continue
            total += (rows * self.sliding_bytes_per_token if m == 0
                      else ctx * self.full_bytes_per_token)
        return total + self.linear_state_bytes

    def resident_bytes(self, ctx, slots):
        return (self.resident_weight_bytes
                + slots * self.expert_stride
                + RUNTIME_OVERHEAD_BYTES
                + self.kv_bytes(ctx))


def gemma_mask():
    mask = [0] * 30
    for i in range(5, 30, 6):
        mask[i] = 1
    return mask


def qwen_mask():
    mask = [2] * 40
    for i in range(3, 40, 4):
        mask[i] = 1
    return mask


# Gemma figures read from the installed pack's manifest.json in this checkout.
GEMMA = Arch("Gemma 4 26B-A4B", gemma_mask(), 8, 256, 2, 512, 1024,
             resident_weight_bytes=1_385_973_080,
             expert_stride=3_358_720,
             installed_bytes=14_291_921_884,
             measured=True)

# Qwen KV geometry is from ArchConfig.qwen36_35B_A3B. The resident split is
# measured: docs/QWEN36_PERFORMANCE.md, 2026-07-31, M5 24 GB, against an
# installed scratch/qwen36.gturbo -- 1.39 GB mapped common weights and 1.13 GB
# of routed-expert slots at 16 per layer (70.6 MB per slot).
#
# That session also corroborates the KV model here to within rounding: it
# reported 84 MB KV and 64 MB recurrent state at 4K, against 80 MiB (83.9 MB)
# and 61 MiB (64.0 MB) predicted. It measured 4K ONLY -- no Qwen long-context
# ladder exists, and RUNTIME_OVERHEAD_BYTES below is Gemma's, carried over.
QWEN = Arch("Qwen 3.6 35B-A3B", qwen_mask(), 2, 256, 2, 256, 0,
            resident_weight_bytes=1_390_000_000,
            expert_stride=70_625_000,
            installed_bytes=19_546_491_213,
            linear={"num_k_heads": 16, "num_v_heads": 32, "key_head_dim": 128,
                    "value_head_dim": 128, "conv_kernel_size": 4},
            measured=True,
            overhead_is_borrowed=True)

DEVICES = [("M4 16 GB", 16 * GIB), ("M5 Max 36 GB", 36 * GIB)]
CONTEXTS = [4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144]

# The five measured Gemma ladder rungs: (prompt tokens, peak footprint MB).
# Source: docs/experiments/summaries/10-long-context.md, M5 Max 36 GB, 16
# expert slots, chunked prefill. The projection must cover every one of them.
MEASURED_GEMMA_RUNGS = [
    (57_344, 3_441),
    (90_112, 4_082),
    (122_880, 4_714),
    (188_416, 6_032),
    (253_952, 7_320),
]

# KV is allocated at the CONTEXT CAP, not at the prompt length. Confirmed on
# 2026-08-28 (M5 Max 36 GB, commit with the family-aware guard): a server
# started with --max-context 262144 and sent a 14-token prompt reached a
# 7,268 MB footprint -- within 52 MB of the 253,952-token rung above.
#
# The practical consequence is that `--max-context` is the memory dial. Sizing
# it to the largest prompt you might ever send costs that memory on every
# request, including trivial ones.
CAP_DRIVEN_ALLOCATION_CHECK = (262_144, 14, 7_268)

# Independent corroboration of the KV model from a real Qwen install, measured
# 2026-07-31 on an M5 24 GB host (docs/QWEN36_PERFORMANCE.md). These are the
# doc's decimal-MB figures at 4K context; the model must reproduce them to
# within 1 MB. This is the only Qwen memory measurement that exists.
QWEN_4K_CORROBORATION = {"kv_excluding_state_mb": 84, "linear_state_mb": 64}

# Share of unified memory available to one process before the system leans on
# swap. Deliberately conservative: the OS and UI need the rest.
USABLE_FRACTION = 0.70


def validate():
    """Check the projection against every measured rung.

    A model that under-predicts is worse than no model, because the whole
    point is telling someone whether a context will fit. Fails loudly.
    """
    print("Validating the projection against the measured Gemma rungs.\n")
    print(f"{'prompt':>9} {'measured':>10} {'projected':>10} {'margin':>9}")
    ok = True
    for prompt, measured_mb in MEASURED_GEMMA_RUNGS:
        projected_mb = GEMMA.resident_bytes(prompt, 16) / MIB
        margin = projected_mb - measured_mb
        if margin < 0:
            ok = False
        print(f"{prompt:>9,} {measured_mb:>9,}M {projected_mb:>9,.0f}M "
              f"{margin:>+8.0f}M")
    print()
    cap, prompt_tokens, measured_mb = CAP_DRIVEN_ALLOCATION_CHECK
    projected_mb = GEMMA.resident_bytes(cap, 16) / MIB
    print(f"Cap-driven allocation check: a {prompt_tokens}-token prompt at "
          f"--max-context {cap:,}")
    print(f"  measured {measured_mb:,} MB vs projected {projected_mb:,.0f} MB "
          f"({projected_mb - measured_mb:+,.0f} MB)")
    print("  KV is sized by the cap, not the prompt.")
    if projected_mb < measured_mb:
        ok = False
    print()

    # Qwen: the KV model is checked against the one real Qwen measurement.
    kv_total = QWEN.kv_bytes(4096)
    kv_only_mb = (kv_total - QWEN.linear_state_bytes) / 1e6
    state_mb = QWEN.linear_state_bytes / 1e6
    exp_kv = QWEN_4K_CORROBORATION["kv_excluding_state_mb"]
    exp_state = QWEN_4K_CORROBORATION["linear_state_mb"]
    print("Qwen 4K corroboration (M5 24 GB, 2026-07-31, real install)")
    print(f"  KV excluding state: measured {exp_kv} MB vs "
          f"modelled {kv_only_mb:.1f} MB")
    print(f"  Recurrent state:    measured {exp_state} MB vs "
          f"modelled {state_mb:.1f} MB")
    if abs(kv_only_mb - exp_kv) > 1 or abs(state_mb - exp_state) > 1:
        ok = False
        print("  MISMATCH: the KV model disagrees with the Qwen measurement.")
    else:
        print("  Agrees within rounding.")
    print()

    if ok:
        print("PASS: the projection covers every measured point.")
    else:
        print("FAIL: the projection under-predicts a measured point. "
              "Raise RUNTIME_OVERHEAD_BYTES.")
    return 0 if ok else 1


def emit(arch, fmt, slots):
    print(f"\n## {arch.name}\n")
    print(f"Installed pack: {arch.installed_bytes / GIB:.2f} GiB "
          f"(most of it streamed, not resident)")
    if arch.measured:
        print(f"Resident weights: {arch.resident_weight_bytes / MIB:,.0f} MiB")
        print(f"Expert cache at {slots} slots: "
              f"{slots * arch.expert_stride / MIB:,.0f} MiB")
        overhead_note = ("MiB (borrowed from Gemma -- unmeasured here)"
                         if arch.overhead_is_borrowed else "MiB (measured)")
        print(f"Runtime overhead: {RUNTIME_OVERHEAD_BYTES / MIB:,.0f} "
              f"{overhead_note}")
    else:
        print("Resident weights / expert stride: NOT MEASURED "
              "(pack not installed in this checkout)")
    print(f"Marginal KV per token: {arch.marginal_kv_bytes_per_token:,} bytes")
    if arch.linear_state_bytes:
        print(f"Fixed linear-attention state: "
              f"{arch.linear_state_bytes / MIB:.0f} MiB (constant in context)")
    print()

    if fmt == "markdown":
        cols = ["Context", "KV"] + (["Total resident"] if arch.measured else [])
        cols += [d for d, _ in DEVICES] if arch.measured else []
        print("| " + " | ".join(cols) + " |")
        print("|" + "---:|" * len(cols))
    else:
        head = f"{'Context':>9} {'KV':>10}"
        if arch.measured:
            head += f" {'Total':>9}  " + "  ".join(f"{d:>13}" for d, _ in DEVICES)
        print(head)

    for ctx in CONTEXTS:
        kv = arch.kv_bytes(ctx)
        label = f"{ctx // 1024}K"
        if not arch.measured:
            if fmt == "markdown":
                print(f"| {label} | {kv / MIB:,.0f} MB |")
            else:
                print(f"{label:>9} {kv / MIB:>9,.0f}M")
            continue
        total = arch.resident_bytes(ctx, slots)
        verdicts = []
        for _, budget in DEVICES:
            headroom = budget * USABLE_FRACTION - total
            verdicts.append(f"fits, {headroom / GIB:.1f} GiB free" if headroom > 0
                            else f"over by {abs(headroom) / GIB:.1f} GiB")
        if fmt == "markdown":
            print(f"| {label} | {kv / MIB:,.0f} MB | {total / GIB:.2f} GiB | "
                  + " | ".join(verdicts) + " |")
        else:
            print(f"{label:>9} {kv / MIB:>9,.0f}M {total / GIB:>8.2f}G  "
                  + "  ".join(f"{v:>13}" for v in verdicts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    parser.add_argument("--slots", type=int, default=16,
                        help="expert cache slots (8, 16, 24, or 32)")
    parser.add_argument("--validate", action="store_true",
                        help="check the projection against the measured rungs")
    args = parser.parse_args()
    if args.validate:
        raise SystemExit(validate())
    print("Resident = weights held + expert slots + runtime overhead + FP16 KV.")
    print("KV assumes chunked prefill (ring enabled), the supported long path.")
    print(f"Verdicts assume {USABLE_FRACTION:.0%} of unified memory is usable.")
    for arch in (GEMMA, QWEN):
        emit(arch, args.format, args.slots)


if __name__ == "__main__":
    main()
