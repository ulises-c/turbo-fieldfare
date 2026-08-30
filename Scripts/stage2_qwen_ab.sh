#!/usr/bin/env bash
# Stage 2B A/B on Qwen 3.6 — expert-cache slots, matched controls.
#
# Methodology mirrors the Gemma expert-slot A/B already in STAGE2_PLAN.md:
#   - two warmup runs per case, discarded, so page-cache state is warm and
#     equal before any measured run (the first Gemma pass was invalidated by
#     page-cache warming aliasing onto run order);
#   - arms interleaved A/B/A/B/... so run order cannot alias the lever;
#   - N measured pairs; every stdout hashed and asserted byte-identical;
#   - per run we capture tok/s, expert-io-await, and unaccounted from the
#     instrumented stderr (TURBO_FIELDFARE_KERNEL_STATS=1 + PHASES=1).
#
# Single branch, single binary — no rebuild sits between runs. Read-only
# w.r.t. the model; writes only under /tmp.

set -uo pipefail

REPO="/Users/ulises/github/turbo-fieldfare-ulises"
BIN="$REPO/.build/release/TurboFieldfareCLI"
MODEL="scratch/qwen36.gturbo"
OUT="/tmp/stage2_qwen"
CASE="${CASE:-short-explanation}"
SEED="${SEED:-20260721}"
PAIRS="${PAIRS:-4}"
WARMUPS="${WARMUPS:-2}"
ARM_A="${ARM_A:-16}"   # default
ARM_B="${ARM_B:-32}"

cd "$REPO" || exit 1
mkdir -p "$OUT"
: > "$OUT/results.csv"
echo "arm,run,tok_s,expert_io_await_ms,unaccounted_ms,sha16" >> "$OUT/results.csv"

run_one() {
  local slots="$1" tag="$2"
  local so="$OUT/${tag}.stdout" se="$OUT/${tag}.stderr"
  TURBO_FIELDFARE_KERNEL_STATS=1 TURBO_FIELDFARE_PHASES=1 "$BIN" \
    --model "$MODEL" \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${CASE}.json" \
    --max-new 128 --max-context 4096 \
    --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$SEED" \
    --expert-cache-slots "$slots" \
    > "$so" 2> "$se"
  local toks await unacc sha
  toks=$(grep -oE 'tok/s=[0-9.]+' "$se" | tail -1 | cut -d= -f2)
  await=$(grep -oE 'expert io await:[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  unacc=$(grep -oE 'unaccounted \(GPU waits\):[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  sha=$(shasum -a 256 < "$so" | cut -c1-16)
  echo "$toks $await $unacc $sha"
}

echo "=== Stage 2B Qwen A/B: case=$CASE seed=$SEED pairs=$PAIRS warmups=$WARMUPS slots ${ARM_A} vs ${ARM_B} ==="
echo "--- warmups (discarded) ---"
for w in $(seq 1 "$WARMUPS"); do
  run_one "$ARM_A" "warm_a_$w" >/dev/null
  run_one "$ARM_B" "warm_b_$w" >/dev/null
  echo "warmup $w done"
done

echo "--- measured (interleaved A/B) ---"
for p in $(seq 1 "$PAIRS"); do
  read -r ta aa ua sa <<<"$(run_one "$ARM_A" "a_$p")"
  echo "A(${ARM_A}) run $p: tok/s=$ta await=$aa unacc=$ua sha=$sa"
  echo "$ARM_A,$p,$ta,$aa,$ua,$sa" >> "$OUT/results.csv"
  read -r tb ab ub sb <<<"$(run_one "$ARM_B" "b_$p")"
  echo "B(${ARM_B}) run $p: tok/s=$tb await=$ab unacc=$ub sha=$sb"
  echo "$ARM_B,$p,$tb,$ab,$ub,$sb" >> "$OUT/results.csv"
done

echo
echo "=== byte-identity check (all stdouts must share one hash) ==="
sort -u <(for f in "$OUT"/a_*.stdout "$OUT"/b_*.stdout; do shasum -a 256 < "$f" | cut -c1-16; done) \
  | { mapfile -t hs; echo "distinct hashes: ${#hs[@]} -> ${hs[*]}"; [[ ${#hs[@]} -eq 1 ]] && echo "IDENTICAL OK" || echo "DIFFERS FAIL"; }

echo
echo "=== results.csv ==="
cat "$OUT/results.csv"
