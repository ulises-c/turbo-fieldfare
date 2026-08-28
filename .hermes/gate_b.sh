#!/usr/bin/env bash
# Gate B — prove the Qwen 3.6 merge did not change Gemma 4's output.
#
# Same model, same frozen prompts, same fixed seeds, greedy-ish sampling:
# fork-main and feat/qwen36-arch must emit byte-identical text. The merge
# rewrote Gemma's hot path (RealForwardRunner +1173/-325), so "the unit
# tests pass" is not evidence about generated tokens. This is.
#
# Non-destructive: builds each branch into its own scratch path, never
# touches scratch/gemma4.gturbo, restores the starting branch on exit.

set -uo pipefail

REPO="/Users/ulises/github/turbo-fieldfare-ulises"
MODEL="scratch/gemma4.gturbo"
OUT="/tmp/gateB"
BASE_REF="origin/fork-main"
HEAD_REF="feat/qwen36-arch"

cd "$REPO" || exit 1

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
cleanup() {
  local current
  current="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  if [[ "$current" != "$START_BRANCH" ]]; then
    echo "[cleanup] restoring branch $START_BRANCH"
    git checkout --quiet "$START_BRANCH" 2>/dev/null
  fi
}
trap cleanup EXIT

# Refuse to run against a dirty tree: a stray edit would be silently
# compiled into one side of the comparison and invalidate the result.
if ! git diff --quiet HEAD; then
  echo "FAIL: working tree is dirty; commit or stash before Gate B." >&2
  exit 1
fi

if [[ ! -d "$MODEL" ]]; then
  echo "FAIL: $MODEL not found." >&2
  exit 1
fi

mkdir -p "$OUT"

# Build one side into its own scratch path and copy the binary out, so
# both binaries coexist and no rebuild happens between generation runs.
build_side() {
  local ref="$1" tag="$2" scratch=".build-gateB-$2"
  echo "=== [$tag] checkout $ref ==="
  git checkout --quiet --detach "$ref" || return 1
  git log --oneline -1
  echo "=== [$tag] building (scratch: $scratch) ==="
  swift build -c release --scratch-path "$scratch" 2>&1 | tail -3 || return 1
  cp "$scratch/release/TurboFieldfareCLI" "$OUT/TurboFieldfareCLI.$tag" || return 1
  echo "[$tag] binary staged at $OUT/TurboFieldfareCLI.$tag"
}

run_cases() {
  local tag="$1" bin="$OUT/TurboFieldfareCLI.$tag"
  mkdir -p "$OUT/$tag"
  for case_seed in \
    short-explanation:20260721 \
    medium-review:20260722 \
    long-synthesis:20260723; do
    local case_id="${case_seed%%:*}" seed="${case_seed##*:}"
    echo "--- [$tag] $case_id (seed $seed) ---"
    "$bin" \
      --model "$MODEL" \
      --messages-file "docs/benchmark-prompts/real-generation-v1/${case_id}.json" \
      --max-new 256 \
      --max-context 4096 \
      --temperature 0.2 \
      --top-k 64 \
      --top-p 0.95 \
      --seed "$seed" \
      > "$OUT/$tag/${case_id}.stdout" \
      2> "$OUT/$tag/${case_id}.stderr"
    echo "[$tag] $case_id exit=$? bytes=$(wc -c < "$OUT/$tag/${case_id}.stdout")"
  done
}

build_side "$BASE_REF" base || { echo "FAIL: base build failed"; exit 1; }
build_side "$HEAD_REF" head || { echo "FAIL: head build failed"; exit 1; }

# Generation runs need the prompt fixtures; use the head checkout's copy.
git checkout --quiet "$HEAD_REF"

echo
echo "############ RUN: base (fork-main) ############"
run_cases base
echo
echo "############ RUN: head (feat/qwen36-arch) ############"
run_cases head

echo
echo "################ GATE B VERDICT ################"
identical=1
for case_id in short-explanation medium-review long-synthesis; do
  if diff -q "$OUT/base/${case_id}.stdout" "$OUT/head/${case_id}.stdout" >/dev/null 2>&1; then
    echo "IDENTICAL  $case_id  ($(shasum -a 256 < "$OUT/head/${case_id}.stdout" | cut -c1-16))"
  else
    identical=0
    echo "DIFFERS    $case_id"
    echo "--- first divergence ---"
    diff "$OUT/base/${case_id}.stdout" "$OUT/head/${case_id}.stdout" | head -20
  fi
done

echo
if [[ $identical -eq 1 ]]; then
  echo "GATE B: PASS — Gemma 4 output is byte-identical across the merge."
else
  echo "GATE B: FAIL — Gemma 4 output changed. This is a blocking defect."
fi
exit $(( identical == 1 ? 0 : 1 ))
