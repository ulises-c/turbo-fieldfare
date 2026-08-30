#!/usr/bin/env python3
"""Agentic multi-turn long-context session: grow to a target context in steps.

WHY THIS EXISTS
---------------
``context_ladder.py`` asks a *cold* question: can the server swallow an N-token
prompt in a single fresh prefill. That is the hardest, least realistic path --
no agent re-prefills its whole history every turn. Attention is causal, so the
KV state at position 256K is identical whether it was built in one 256K prefill
or across sixteen 16K turns. The realistic path is the second one: a live
session whose context grows turn by turn as tool results arrive, with the KV
retained (``--prompt-cache-mode single-prefix``) so each turn only prefills its
*new* tokens.

This harness simulates that. One server stays alive for the whole session. Each
turn appends a new user message (a ~``--step-tokens`` "tool result" of filler)
and the model answers; that answer becomes history for the next turn, exactly
as the prompt cache requires (it keys on the model's own generated tokens, so
we replay each response verbatim -- pasting arbitrary assistant text would
diverge the prefix and force a full re-prefill).

WHAT IT MEASURES, PER TURN
--------------------------
- context depth (cumulative prompt tokens)
- cached_tokens (prefix reused) vs new tokens prefilled this turn
- time to first token (TTFT) -- the agentic latency that actually matters
- decode tokens/second at this depth (the cold ladder can't see this: it takes
  a one-token completion)
- turn wall time, peak footprint, peak Metal, min free memory, swap

RECALL AS A BY-PRODUCT
----------------------
A distinctive needle is planted in the FIRST turn's context. At each checkpoint
we ask for it back, so the run doubles as "does retrieval survive as the
session grows to 256K" -- recall at depth, across real turns, not a cold probe.

Because the needle lives at the very start, it sits at fractional depth ~0.0 as
the session grows: the hardest position for a sliding-window model, since it
must survive on the full-attention layers alone across the entire span.

USAGE
-----
    python3 Scripts/context_session.py \\
      --model scratch/qwen36.gturbo --model-id qwen3.6-35b-a3b \\
      --step-tokens 16384 --target 262144 \\
      --checkpoints 65536,98304,131072,196608,262144
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "scratch" / "gemma4.gturbo"
DEFAULT_OUT = ROOT / "benchmark-results" / "context-session"

# A distinctive, verbatim-checkable needle planted at the very start of the
# session context, plus the question we re-ask at each checkpoint.
NEEDLE_CODE = "74391-QM"
NEEDLE = f"The Aurora Vault access code is {NEEDLE_CODE}."
QUESTION = "What is the Aurora Vault access code? Answer with only the code."
# Filler token == one "word " for both supported tokenizers (measured 1.000).
FILLER_WORD = "word"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def wait_ready(proc: subprocess.Popen[str], logs: list[str],
               ready: threading.Event) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logs.append(line.rstrip())
        if "TurboFieldfareServer ready" in line:
            ready.set()


def model_id_from_logs(logs: list[str]) -> str | None:
    for line in logs:
        if "TurboFieldfareServer ready" not in line:
            continue
        for token in line.split():
            if token.startswith("model="):
                return token[len("model="):]
    return None


def start_server(options: argparse.Namespace
                 ) -> tuple[subprocess.Popen[str], list[str]]:
    command = [
        str(options.server),
        "--model", str(options.model),
        "--port", str(options.port),
        "--max-context", str(options.max_context),
        "--prompt-cache-mode", "single-prefix",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs: list[str] = []
    ready = threading.Event()
    threading.Thread(target=wait_ready, args=(proc, logs, ready),
                     daemon=True).start()
    if not ready.wait(options.ready_timeout):
        proc.terminate()
        raise RuntimeError(f"server did not become ready: {logs[-10:]}")
    return proc, logs


def memory_sample(pid: int) -> dict:
    sample: dict[str, object] = {}
    try:
        text = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        match = re.search(r"free percentage: (\d+)%", text)
        if match:
            sample["memory_free_pct"] = int(match.group(1))
    except Exception:
        pass
    try:
        text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        match = re.search(r"used = ([0-9.]+)([MG])", text)
        if match:
            value = float(match.group(1))
            sample["swap_used_mb"] = value * (1024 if match.group(2) == "G" else 1)
    except Exception:
        pass
    try:
        text = subprocess.check_output(["footprint", "-p", str(pid)], text=True)
        match = re.search(r"Footprint:\s+([0-9.]+) MB", text)
        if match:
            sample["footprint_mb"] = float(match.group(1))
        match = re.search(
            r"(\d+(?:\.\d+)?) MB\s+0 B\s+0 B\s+\d+\s+IOAccelerator \(graphics\)",
            text)
        if match:
            sample["metal_mb"] = float(match.group(1))
    except Exception:
        pass
    return sample


def filler_block(word_count: int) -> str:
    return " ".join([FILLER_WORD] * word_count)


def post_streaming(options: argparse.Namespace, messages: list[dict],
                   max_tokens: int) -> dict:
    """One streaming chat completion. Returns text, usage, TTFT, decode timing.

    Streaming is what exposes TTFT: the first SSE chunk carrying content marks
    first-token latency, which for a warm session is dominated by the new
    turn's prefill, not the whole context.
    """
    payload = json.dumps({
        "model": options.model_id,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{options.port}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    started = time.time()
    first_token_at: float | None = None
    pieces: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=options.request_timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.time()
                    pieces.append(piece)
    finished = time.time()
    text = "".join(pieces)
    completion_tokens = usage.get("completion_tokens") or 0
    ttft = (first_token_at - started) if first_token_at else None
    decode_seconds = (finished - first_token_at) if first_token_at else None
    decode_tok_s = (
        (completion_tokens - 1) / decode_seconds
        if decode_seconds and decode_seconds > 0 and completion_tokens > 1
        else None)
    return {
        "text": text,
        "usage": usage,
        "wall_seconds": round(finished - started, 3),
        "ttft_seconds": round(ttft, 3) if ttft else None,
        "decode_seconds": round(decode_seconds, 3) if decode_seconds else None,
        "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else None,
    }


def server_completion_timing(logs: list[str]) -> dict:
    """Pull pp/cached from the most recent server completion line."""
    line = next((x for x in reversed(logs) if " completed in " in x), None)
    if not line:
        return {}
    out: dict[str, object] = {}
    m = re.search(r"prompt=(\d+) cached=(\d+) completion=(\d+)", line)
    if m:
        out["server_prompt_tokens"] = int(m.group(1))
        out["server_cached_tokens"] = int(m.group(2))
        out["server_completion_tokens"] = int(m.group(3))
    m = re.search(r"pp=([0-9.]+)s pp_tok_s=([0-9.]+)", line)
    if m:
        out["pp_seconds"] = float(m.group(1))
        out["pp_tokens_per_second"] = float(m.group(2))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic multi-turn long-context session simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server", type=Path,
                        default=ROOT / ".build/release/TurboFieldfareServer")
    parser.add_argument("--model-id", default=None,
                        help="Defaults to the id the server reports at ready.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-context", type=int, default=262_144)
    parser.add_argument("--target", type=int, default=262_144,
                        help="Grow the session until context reaches this.")
    parser.add_argument("--step-tokens", type=int, default=16_384,
                        help="Approx new context tokens added per turn "
                             "(a 'tool result' arriving). 16K ~ a chunky file.")
    parser.add_argument("--checkpoints", default="65536,98304,131072,196608,262144",
                        help="Context depths at which to re-ask the needle and "
                             "sample memory. Comma-separated token counts.")
    parser.add_argument("--answer-tokens", type=int, default=32,
                        help="max_completion_tokens for the model's reply.")
    parser.add_argument("--request-timeout", type=float, default=18_000)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    options = parser.parse_args(argv)
    options.checkpoints = sorted(
        int(p) for p in options.checkpoints.split(",") if p.strip())
    return options


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if not options.model.is_dir():
        print(f"missing model directory: {options.model}", file=sys.stderr)
        return 2
    if not options.server.is_file():
        print(f"missing server binary: {options.server}\n"
              "build it with: swift build -c release --product TurboFieldfareServer",
              file=sys.stderr)
        return 2
    options.commit = git_commit()
    options.out.mkdir(parents=True, exist_ok=True)
    stem = options.model.name.replace(".gturbo", "")
    jsonl = options.out / f"session-{stem}.jsonl"

    proc, logs = start_server(options)
    if options.model_id is None:
        served = model_id_from_logs(logs)
        if served is None:
            proc.terminate()
            raise RuntimeError("could not read model= from ready line")
        options.model_id = served
        print(f"model id: {served} (from the server)", flush=True)
    print(f"commit {options.commit}\nmodel {options.model}\n"
          f"step {options.step_tokens} target {options.target}\n"
          f"checkpoints {options.checkpoints}\nappending to {jsonl}", flush=True)

    # The conversation. Turn 1 seeds the needle + first filler block; each later
    # turn appends a filler "tool result" and the model's verbatim prior answer.
    messages: list[dict] = []
    records: list[dict] = []
    approx_depth = 0
    turn = 0
    # First turn also carries the needle and the standing instruction.
    step_words = options.step_tokens
    pending_checkpoints = list(options.checkpoints)

    def next_is_checkpoint(depth_after: int) -> bool:
        return bool(pending_checkpoints) and depth_after >= pending_checkpoints[0]

    try:
        while approx_depth < options.target:
            turn += 1
            # Ask the needle question on a checkpoint turn; otherwise a trivial
            # ack keeps the turn cheap but real.
            #
            # Clamp the step so the projected server-side context never exceeds
            # --max-context. `approx_depth` is REAL tokens but `step_words` is a
            # filler WORD count, and filler_block() + the chat template emit more
            # tokens than words. Leave headroom for the answer budget plus a
            # tokenization-overhead margin, or the final turn overshoots the cap
            # and the server returns HTTP 400 (as happened at the 256K rung).
            answer_budget = options.answer_tokens if next_is_checkpoint(
                min(options.target, approx_depth + step_words)) else 4
            OVERHEAD_MARGIN = 512  # chat-template + BPE slack, conservative
            room = (options.max_context - approx_depth
                    - answer_budget - OVERHEAD_MARGIN)
            if room <= 0:
                # No headroom left under the cap: stop cleanly rather than 400.
                turn -= 1
                break
            this_step = min(step_words, room)
            depth_after = min(options.target, approx_depth + this_step)
            checkpoint = next_is_checkpoint(depth_after)
            if turn == 1:
                content = (f"{NEEDLE} " + filler_block(this_step)
                           + f"\n\nAcknowledge you have received the context.")
            else:
                body = filler_block(this_step)
                if checkpoint:
                    content = f"{body}\n\n{QUESTION}"
                else:
                    content = f"{body}\n\nReply 'ack' only."
            messages.append({"role": "user", "content": content})

            max_tokens = options.answer_tokens if checkpoint else 4
            result = post_streaming(options, messages, max_tokens)
            # Replay the model's own answer verbatim -- the cache keys on it.
            messages.append({"role": "assistant", "content": result["text"]})

            timing = server_completion_timing(logs)
            mem = memory_sample(proc.pid)
            usage = result["usage"]
            depth = usage.get("prompt_tokens", approx_depth)
            approx_depth = depth + (usage.get("completion_tokens") or 0)

            # New tokens actually prefilled this turn = total prompt minus the
            # KV prefix reused from the prior turn. This is the real per-turn
            # cost the whole session mode exists to expose; the raw server
            # prompt count is the full context and would hide the cache win.
            server_prompt = timing.get("server_prompt_tokens")
            server_cached = timing.get("server_cached_tokens")
            new_prefill = (
                server_prompt - server_cached
                if server_prompt is not None and server_cached is not None
                else None)

            found = None
            if checkpoint:
                found = NEEDLE_CODE in result["text"]
                if pending_checkpoints:
                    pending_checkpoints.pop(0)

            record = {
                "turn": turn,
                "checkpoint": checkpoint,
                "context_tokens": depth,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "new_prefill_tokens": new_prefill,
                "server_cached_tokens": timing.get("server_cached_tokens"),
                "pp_seconds": timing.get("pp_seconds"),
                "pp_tokens_per_second": timing.get("pp_tokens_per_second"),
                "ttft_seconds": result["ttft_seconds"],
                "decode_tok_s": result["decode_tok_s"],
                "completion_tokens": usage.get("completion_tokens"),
                "turn_wall_seconds": result["wall_seconds"],
                "footprint_mb": mem.get("footprint_mb"),
                "metal_mb": mem.get("metal_mb"),
                "memory_free_pct": mem.get("memory_free_pct"),
                "swap_used_mb": mem.get("swap_used_mb"),
                "needle_found": found,
                "answer": result["text"][:120] if checkpoint else None,
                "source_commit": options.commit,
            }
            records.append(record)
            with jsonl.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            tag = "CHECKPOINT" if checkpoint else "turn"
            print(f"  {tag} {turn}: ctx={depth:,} "
                  f"cached={record['cached_tokens']} "
                  f"new_pp={record['new_prefill_tokens']} "
                  f"ttft={record['ttft_seconds']}s "
                  f"pp_tok_s={record['pp_tokens_per_second']} "
                  f"decode_tok_s={record['decode_tok_s']} "
                  + (f"needle={'HIT' if found else 'MISS'} " if checkpoint else "")
                  + f"free={record['memory_free_pct']}%", flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)

    (options.out / f"session-{stem}.json").write_text(
        json.dumps({"commit": options.commit, "model_id": options.model_id,
                    "step_tokens": options.step_tokens,
                    "turns": records}, indent=2) + "\n")
    checkpoints = [r for r in records if r["checkpoint"]]
    misses = sum(1 for r in checkpoints if r["needle_found"] is False)
    print(f"\n{len(records)} turns, {len(checkpoints)} checkpoints, "
          f"{misses} needle miss(es)", flush=True)
    return 0 if misses == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
