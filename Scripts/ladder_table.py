#!/usr/bin/env python3
"""Render a per-context-stage table from context_ladder.py output.

Reads the per-rung ``<label>K.json`` artifacts (or the merged ``results.json``)
a ladder run wrote and prints one row per context stage: the columns the
long-context experiment note uses (prompt tokens, end-to-end time, prefill
time and throughput, peak footprint, peak Metal, minimum free memory, swap,
result). Emits both a GitHub-flavored Markdown table and a compact JSON blob so
the numbers can be pasted into docs or diffed between models.

Usage::

    python3 Scripts/ladder_table.py benchmark-results/context-ladder-qwen36
    python3 Scripts/ladder_table.py benchmark-results/context-ladder --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rungs(directory: Path) -> list[dict]:
    """Prefer the merged aggregate; fall back to per-rung files."""
    aggregate = directory / "results.json"
    if aggregate.is_file():
        try:
            rows = json.loads(aggregate.read_text())
            if rows:
                return sorted(rows, key=lambda r: r.get("max_context", 0))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: unreadable {aggregate}: {exc}", file=sys.stderr)
    rows = []
    for path in sorted(directory.glob("*K.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
    return sorted(rows, key=lambda r: r.get("max_context", 0))


def classify(rung: dict) -> str:
    if rung.get("client_status") != "completed":
        return rung.get("client_error", "error")
    if rung.get("cancel_log"):
        return "cancelled"
    if rung.get("finish_reason") and rung.get("finish_reason") != "length":
        return f"finish={rung['finish_reason']}"
    if rung.get("prompt_tokens") is None:
        return "no completion log"
    return "completed"


def row_for(rung: dict) -> dict:
    return {
        "context_k": rung.get("label_k"),
        "max_context": rung.get("max_context"),
        "prompt_tokens": rung.get("prompt_tokens"),
        "cached_tokens": rung.get("cached_tokens"),
        "e2e_seconds": rung.get("server_total_seconds"),
        "prefill_seconds": rung.get("pp_seconds"),
        "prefill_tok_s": rung.get("pp_tokens_per_second"),
        "peak_footprint_mb": rung.get("peak_footprint_mb"),
        "peak_metal_mb": rung.get("peak_ioaccelerator_mb"),
        "min_free_mem_pct": rung.get("minimum_memory_free_pct"),
        "peak_swap_mb": rung.get("peak_swap_used_mb"),
        "result": classify(rung),
    }


def fmt(value: object, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}{suffix}" if suffix == "s" else f"{value:,.0f}{suffix}"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def fmt_rate(value: object) -> str:
    """Prefill throughput keeps three decimals to match the experiment note."""
    if value is None:
        return "-"
    return f"{float(value):,.3f}"


def markdown(rows: list[dict], commit: str | None) -> str:
    header = ("| Context cap | Actual prompt | E2E time | Prefill | Prefill tok/s "
              "| Peak footprint | Peak Metal | Min free mem | Swap | Result |")
    divider = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for r in rows:
        lines.append(
            f"| {r['context_k']}K "
            f"| {fmt(r['prompt_tokens'])} "
            f"| {fmt(r['e2e_seconds'], 's')} "
            f"| {fmt(r['prefill_seconds'], 's')} "
            f"| {fmt_rate(r['prefill_tok_s'])} "
            f"| {fmt(r['peak_footprint_mb'], ' MB')} "
            f"| {fmt(r['peak_metal_mb'], ' MB')} "
            f"| {fmt(r['min_free_mem_pct'], '%')} "
            f"| {fmt(r['peak_swap_mb'], ' MB')} "
            f"| {r['result']} |")
    if commit:
        lines.append("")
        lines.append(f"_Source commit: `{commit}`_")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path,
                        help="Ladder output dir (holds <label>K.json / results.json).")
    parser.add_argument("--json", action="store_true",
                        help="Emit the extracted rows as JSON instead of Markdown.")
    options = parser.parse_args(argv)

    if not options.directory.is_dir():
        parser.error(f"not a directory: {options.directory}")
    rungs = load_rungs(options.directory)
    if not rungs:
        print(f"no ladder rungs found under {options.directory}", file=sys.stderr)
        return 1
    rows = [row_for(r) for r in rungs]
    commit = next((r.get("source_commit") for r in rungs if r.get("source_commit")), None)

    if options.json:
        print(json.dumps({"commit": commit, "rungs": rows}, indent=2))
    else:
        print(markdown(rows, commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
