#!/usr/bin/env bash
# Peak-RSS check: does --expert-cache-slots 32 cost materially more resident
# memory than the default 16 on Qwen 3.6? One measured run each (after a
# discarded warmup), /usr/bin/time -l peak footprint. Same prompt/seed.
set -uo pipefail
REPO="/Users/ulises/github/turbo-fieldfare-ulises"
BIN="$REPO/.build/release/TurboFieldfareCLI"
MODEL="scratch/qwen36.gturbo"
OUT="/tmp/stage2_qwen_rss"
cd "$REPO" || exit 1
mkdir -p "$OUT"

measure() {
  local slots="$1"
  /usr/bin/time -l "$BIN" \
    --model "$MODEL" \
    --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
    --max-new 128 --max-context 4096 \
    --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 \
    --expert-cache-slots "$slots" \
    > "$OUT/slots_${slots}.stdout" 2> "$OUT/slots_${slots}.time"
  local rss
  rss=$(grep 'maximum resident set size' "$OUT/slots_${slots}.time" | grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ')
  echo "slots=$slots peak_rss_bytes=$rss peak_rss_gib=$(echo "scale=3; $rss/1073741824" | bc)"
}

# discarded warmups
measure 16 >/dev/null; measure 32 >/dev/null
echo "=== measured peak RSS ==="
measure 16
measure 32
echo "=== output identity ==="
shasum -a 256 "$OUT/slots_16.stdout" "$OUT/slots_32.stdout" | cut -c1-16
