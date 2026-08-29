#!/usr/bin/env python3
"""Sweep the long-context ladder for retrieval recall AND fresh prefill timing.

Two open questions are answered by the same runs, so they share one sweep
rather than prefilling the ladder twice:

1. **Retrieval.** ``docs/experiments/summaries/10-long-context.md`` measured
   admission (does the server accept and prefill the prompt) but recall was
   only ever probed at 1,543 and 14,043 tokens. Between 57K and 254K nothing
   was measured, and a degraded long context returns HTTP 200 exactly like a
   healthy one. Gemma 4 gives only 5 of its 30 layers full attention, so this
   is where degradation would appear if it appears at all.

2. **Timing staleness.** Those recorded prefill columns predate the merge of
   "Speed up prefill on pre-Apple10 Macs" (#159), which rewrote the very path
   they measure. They were also recorded across several commits with the 64K
   baseline taken last, so they are not a matched set. Every probe here runs
   on one binary and records ``pp_seconds`` and ``pp_tokens_per_second``,
   which gives a matched replacement as a by-product of the recall sweep.

Each rung starts its own server and stops it before the next, so only one
model process is ever alive. Results append to a JSONL file after every probe,
so an interrupted sweep keeps everything it already measured.

Filler sizing: measured at 14,000 words -> 14,043 tokens (1.0031 tokens/word).
Each rung targets the same prompt length the memory ladder used, leaving the
8,192-token reserve the recorded run used.

Usage::

    python3 Scripts/context_sweep.py                 # full ladder
    python3 Scripts/context_sweep.py --rungs 65536   # one rung
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (context cap, filler words) -- filler chosen so the prompt lands on the same
# token count the memory ladder recorded for that rung.
RUNGS: dict[int, int] = {
    65_536: 57_000,
    98_304: 89_800,
    131_072: 122_500,
    196_608: 187_800,
    262_144: 253_100,
}


def run_rung(context: int, filler: int, depths: str, out: Path,
             jsonl: Path) -> tuple[int, list[dict]]:
    command = [
        sys.executable, str(ROOT / "Scripts" / "context_retrieval.py"),
        "--model", str(ROOT / "scratch" / "gemma4.gturbo"),
        "--max-context", str(context),
        "--filler-words", str(filler),
        "--depths", depths,
        "--out", str(out),
    ]
    print(f"\n=== {context // 1024}K cap, {filler:,} filler words ===",
          flush=True)
    started = time.time()
    proc = subprocess.run(command, text=True, capture_output=True)
    records: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            if line:
                print(line, flush=True)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
        with jsonl.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"  depth {record.get('depth')}: "
              f"found={record.get('found')} "
              f"tokens={record.get('prompt_tokens')} "
              f"pp={record.get('pp_seconds')}s "
              f"pp_tok_s={record.get('pp_tokens_per_second')} "
              f"cached={record.get('cached_tokens')}", flush=True)
    if proc.returncode not in (0, 1):
        print(f"  rung failed rc={proc.returncode}\n{proc.stderr[-2000:]}",
              flush=True)
    print(f"  rung wall time {time.time() - started:,.0f}s", flush=True)
    return proc.returncode, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", default="",
                        help="Comma-separated context caps. Default: all.")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "benchmark-results" / "context-retrieval")
    options = parser.parse_args(argv)

    if options.rungs:
        wanted = [int(part) for part in options.rungs.split(",") if part.strip()]
        unknown = [rung for rung in wanted if rung not in RUNGS]
        if unknown:
            parser.error(f"unknown rungs {unknown}; known: {sorted(RUNGS)}")
    else:
        wanted = sorted(RUNGS)

    options.out.mkdir(parents=True, exist_ok=True)
    jsonl = options.out / "sweep.jsonl"
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    print(f"commit {commit}\nrungs {wanted}\ndepths {options.depths}\n"
          f"appending to {jsonl}", flush=True)

    everything: list[dict] = []
    for context in wanted:
        _, records = run_rung(context, RUNGS[context], options.depths,
                              options.out, jsonl)
        everything.extend(records)

    print("\n=== summary ===", flush=True)
    print(f"{'rung':>8} {'tokens':>9} {'recall':>8} {'pp_tok_s':>10}",
          flush=True)
    misses = 0
    for context in wanted:
        rows = [r for r in everything if r.get("max_context") == context]
        if not rows:
            continue
        found = sum(1 for r in rows if r.get("found"))
        misses += len(rows) - found
        rates = [r["pp_tokens_per_second"] for r in rows
                 if r.get("pp_tokens_per_second")]
        mean = sum(rates) / len(rates) if rates else float("nan")
        tokens = rows[0].get("prompt_tokens")
        print(f"{context // 1024:>7}K {tokens:>9,} {found:>4}/{len(rows):<3} "
              f"{mean:>10.2f}", flush=True)
    print(f"\n{misses} miss(es) across {len(everything)} probes", flush=True)
    return 0 if misses == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
