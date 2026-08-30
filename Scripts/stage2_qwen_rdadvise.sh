#!/usr/bin/env bash
# Stage 2C rdadvise A/B on Qwen 3.6 — read-advice policy, matched controls.
#
# The Gemma rdadvise pass in STAGE2_PLAN.md was order-confounded (arms run
# back to back) and only recorded as "no effect observed". This re-measures
# it cleanly with the same treatment as the slot A/B: warmups discarded,
# arms interleaved A/B/A/B, byte-identical stdout asserted, per-run tok/s +
# expert-io-await + unaccounted captured.
#
# ARM_A defaults to the shipping default (off); ARM_B to adaptive (the most
# aggressive policy). Single binary, no rebuild between runs.

set -uo pipefail
REPO="/Users/ulises/github/turbo-fieldfare-ulises"
BIN="$REPO/.build/release/TurboFieldfareCLI"
MODEL="scratch/qwen36.gturbo"
OUT="/tmp/stage2_qwen_rdadvise"
CASE="${CASE:-short-explanation}"
SEED="${SEED:-20260721}"
PAIRS="${PAIRS:-4}"
WARMUPS="${WARMUPS:-2}"
ARM_A="${ARM_A:-off}"
ARM_B="${ARM_B:-adaptive}"

cd "$REPO" || exit 1
mkdir -p "$OUT"
: > "$OUT/results.csv"
echo "arm,run,tok_s,expert_io_await_ms,unaccounted_ms,sha16" >> "$OUT/results.csv"

run_one() {
  local policy="$1" tag="$2"
  local so="$OUT/${tag}.stdout" se="$OUT/${tag}.stderr"
  TURBO_FIELDFARE_KERNEL_STATS=1 TURBO_FIELDFARE_PHASES=1 "$BIN" \
    --model "$MODEL" \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${CASE}.json" \
    --max-new 128 --max-context 4096 \
    --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$SEED" \
    --rdadvise "$policy" \
    > "$so" 2> "$se"
  local toks await unacc sha
  toks=$(grep -oE 'tok/s=[0-9.]+' "$se" | tail -1 | cut -d= -f2)
  await=$(grep -oE 'expert io await:[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  unacc=$(grep -oE 'unaccounted \(GPU waits\):[[:space:]]+[0-9.]+' "$se" | tail -1 | grep -oE '[0-9.]+$')
  sha=$(shasum -a 256 < "$so" | cut -c1-16)
  echo "$toks $await $unacc $sha"
}

echo "=== Stage 2C Qwen rdadvise A/B: case=$CASE seed=$SEED pairs=$PAIRS warmups=$WARMUPS policy ${ARM_A} vs ${ARM_B} ==="
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
echo "=== byte-identity check ==="
sort -u <(for f in "$OUT"/a_*.stdout "$OUT"/b_*.stdout; do shasum -a 256 < "$f" | cut -c1-16; done) \
  | { mapfile -t hs; echo "distinct hashes: ${#hs[@]} -> ${hs[*]}"; [[ ${#hs[@]} -eq 1 ]] && echo "IDENTICAL OK" || echo "DIFFERS FAIL"; }
echo
echo "=== results.csv ==="
cat "$OUT/results.csv"
