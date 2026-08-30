#!/usr/bin/env bash
# Stage 2B baseline decode profile on Qwen 3.6 — the three frozen prompts,
# default settings, instrumentation on. Reproduces the Gemma profile-table
# shape (prefill / decode / tok/s / expert-io-await / unaccounted / kernel
# total) for the Qwen family. One warmup per case discarded.

set -uo pipefail
REPO="/Users/ulises/github/turbo-fieldfare-ulises"
BIN="$REPO/.build/release/TurboFieldfareCLI"
MODEL="scratch/qwen36.gturbo"
OUT="/tmp/stage2_qwen_profile"
cd "$REPO" || exit 1
mkdir -p "$OUT"

run() {
  local case_id="$1" seed="$2" tag="$3"
  TURBO_FIELDFARE_KERNEL_STATS=1 TURBO_FIELDFARE_PHASES=1 "$BIN" \
    --model "$MODEL" \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
    --max-new 128 --max-context 4096 \
    --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$seed" \
    > "$OUT/${tag}.stdout" 2> "$OUT/${tag}.stderr"
}

echo "prompt | prefill | decode | tok/s | expert_io_await | unaccounted | attn_router | kernel_total"
for cs in short-explanation:20260721 medium-review:20260722 long-synthesis:20260723; do
  cid="${cs%%:*}"; seed="${cs##*:}"
  run "$cid" "$seed" "warm_$cid" >/dev/null 2>&1   # discarded warmup
  run "$cid" "$seed" "$cid"
  se="$OUT/${cid}.stderr"
  prefill=$(grep -oE 'prefill=[0-9]+tok/[0-9.]+s' "$se" | tail -1)
  decode=$(grep -oE 'decode=[0-9.]+s' "$se" | tail -1)
  toks=$(grep -oE 'tok/s=[0-9.]+' "$se" | tail -1 | cut -d= -f2)
  await=$(grep -oE 'expert io await:[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  unacc=$(grep -oE 'unaccounted \(GPU waits\):[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  ar=$(grep -oE 'role=attention_router gpu_ms=[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  kt=$(grep -oE 'role=total gpu_ms=[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  echo "$cid | $prefill | $decode | $toks | ${await}ms | ${unacc}ms | ${ar}ms | ${kt}ms"
done
