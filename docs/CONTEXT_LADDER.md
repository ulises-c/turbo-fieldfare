# Context ladder experiment

This is an exploratory context-capacity and resource experiment, not a community
speed benchmark. The prompt uses repeated `word` tokens to reach controlled
lengths; repeated prompts can produce unusually stable expert routing and should
not be used as a quality or general-purpose performance claim.

## Protocol

- Hardware: Apple M5 Max MacBook Pro (18 CPU cores), 36 GB unified memory
- OS: macOS 26.5.2 (25F84)
- Swift: Apple Swift 6.3.3
- Runtime: release `TurboFieldfareServer`, source base `fork-main` at `c0354d7`
- Model: `gemma-4-26b-a4b-it-ulises` runtime ID, Gemma 4 26B-A4B 4-bit weight pack
- Sampling: temperature 0, one requested output token
- Uniform reserve: 8,192 tokens at every context level
- Prompt targets: 57,344 / 90,112 / 122,880 / 188,416 / 253,952 tokens
- Memory samples: every 10 seconds using `memory_pressure`, `footprint`, `ps`, and `vm.swapusage`
- PP/TG: server log fields `pp`, `pp_tok_s`, `tg`, and `tg_tok_s`

The 8,192-token reserve leaves space for the chat-template overhead and one
completion token. It is uniform across all five levels.

## Why 256K

The [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
lists a 256K-token context window for the 26B A4B model. The server therefore
tests the model's native maximum rather than extrapolating beyond the model's
published context length. With the uniform 8,192-token reserve, the 256K rung
uses a `--max-context` value of 262,144 and targets 253,952 prompt tokens.

## Testing methodology

`Scripts/context_ladder.py` orchestrates each rung as an isolated run:

1. It starts a fresh release `TurboFieldfareServer` subprocess on loopback
   (`127.0.0.1:8080`) with the selected `--max-context` value and waits for the
   ready message.
2. It starts a sampler that records `memory_pressure`, `vm.swapusage`,
   `footprint`, and `ps` data every 10 seconds.
3. It constructs the request body and posts one OpenAI-compatible chat request
   to `/v1/chat/completions`.
4. It records the server-reported usage, completion log, prefill/decode timing,
   and resource samples, then sends SIGINT to the server process it launched.

The context text is deliberately controlled rather than semantic: the harness
creates `"word " * words`, where
`words = max_context - 8,192 - 13`. The 13-token allowance is the measured
Gemma chat-template overhead. The server's `usage.prompt_tokens` is authoritative
for the final count; it reports 57,344, 90,112, 122,880, 188,416, and 253,952
tokens for the five runs. This makes the ladder reproducible, but repeated words
can produce more stable routing than real agent prompts.

A rung passes when the client receives HTTP 200, the server reports the intended
prompt and one-token completion, the completion log contains the timing fields,
and the request is not rejected, timed out, cancelled, truncated by the client,
or terminated by OOM. Resource pressure is reported separately rather than used
as a hidden pass/fail threshold. The one-token output isolates prompt prefill and
memory behavior; it is not a useful decode-throughput benchmark.

## Timing trend

End-to-end time is broadly linear with prompt length over these five points but
not linear at a constant per-token cost. Relative to 64K, prompt length grows
4.43x at 256K while E2E time grows 8.13x. A least-squares fit of E2E seconds to
actual prompt tokens has `R² = 0.996`, but time per prompt token rises from
4.118 ms at 64K to 7.563 ms at 256K. The incremental cost also rises from
6.960 seconds per additional 1K tokens between 64K and 96K to 9.597 seconds
per 1K between 192K and 256K. Prefill throughput therefore declines as context
grows, rather than remaining constant.

## Results

| Context | Prompt | E2E server time | PP time | PP tok/s | TG time | Peak footprint | Peak Metal | Min free unified memory | Swap | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64K | 57,344 | 236.158s | 236.000s | 242.983 | 0.001s* | 3,441 MB | 1,537 MB | 76% | 0 MB | completed |
| 96K | 90,112 | 464.218s | 463.937s | 194.233 | 0.001s* | 4,082 MB | 2,177 MB | 76% | 0 MB | completed |
| 128K | 122,880 | 701.019s | 700.724s | 175.361 | 0.001s* | 4,714 MB | 2,817 MB | 75% | 0 MB | completed |
| 192K | 188,416 | 1,291.620s | 1,290.896s | 145.958 | 0.001s* | 6,032 MB | 4,097 MB | 68% | 0 MB | completed |
| 256K | 253,952 | 1,920.565s | 1,919.868s | 132.276 | 0.001s* | 7,320 MB | 5,377 MB | 62% | 0 MB | completed |

`*` The TG values in this pass are not meaningful throughput measurements: the
probe deliberately requested only one output token to isolate long-context
prefill and memory behavior. Run a separate fixed-output decode pass, such as
128 requested tokens, before making TG performance claims.

The 256K request completed after the client timeout was raised to 60 minutes.
It used 1,920.565 seconds end-to-end (1,919.868 seconds of prefill) and had no
observed swap use, OOM, context-length rejection, or client cancellation.

The new 64K baseline completed in 236.158 seconds at 242.983 prefill tokens per
second, with a 3,441 MB peak footprint and 1,537 MB peak Metal allocation. From
the 64K baseline to 256K, the measured prompt grew from 57,344 to 253,952
tokens, prefill throughput declined from 242.983 to 132.276 tokens per second,
peak footprint grew from 3,441 MB to 7,320 MB, and minimum free memory declined
from 76% to 62%. Neither endpoint used swap.

Raw JSON samples are written by `Scripts/context_ladder.py` to the local,
ignored `benchmark-results/context-ladder/` directory.
