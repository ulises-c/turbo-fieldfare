# Context ladder experiment

This is an exploratory context-capacity and resource experiment, not a community
speed benchmark. The prompt uses repeated `word` tokens to reach controlled
lengths; repeated prompts can produce unusually stable expert routing and should
not be used as a quality or general-purpose performance claim.

## Protocol

- Hardware: Apple M5 Max MacBook Pro, 36 GB unified memory
- OS: macOS 26.5.2
- Runtime: release `TurboFieldfareServer`
- Model: `gemma-4-26b-a4b-it-ulises`
- Sampling: temperature 0, one requested output token
- Uniform reserve: 8,192 tokens at every context level
- Prompt targets: 90,112 / 122,880 / 188,416 / 253,952 tokens
- Memory samples: every 10 seconds using `memory_pressure`, `footprint`, `ps`, and `vm.swapusage`
- PP/TG: server log fields `pp`, `pp_tok_s`, `tg`, and `tg_tok_s`

The 8,192-token reserve leaves space for the chat-template overhead and one
completion token. It is uniform across all four levels.

## Results

| Context | Prompt | E2E server time | PP time | PP tok/s | TG time | Peak footprint | Peak Metal | Min free unified memory | Swap | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 96K | 90,112 | 464.218s | 463.937s | 194.233 | 0.001s* | 4,082 MB | 2,177 MB | 76% | 0 MB | completed |
| 128K | 122,880 | 701.019s | 700.724s | 175.361 | 0.001s* | 4,714 MB | 2,817 MB | 75% | 0 MB | completed |
| 192K | 188,416 | 1,291.620s | 1,290.896s | 145.958 | 0.001s* | 6,032 MB | 4,097 MB | 68% | 0 MB | completed |
| 256K | 253,952 | 30-minute bound removed; rerun with a 60-minute client timeout | not yet completed | — | — | 7,323 MB | 5,377 MB | 62% | 0 MB | prior 30-minute client timeout was too short for the measured PP curve |

`*` The TG values in this pass are not meaningful throughput measurements: the
probe deliberately requested only one output token to isolate long-context
prefill and memory behavior. Run a separate fixed-output decode pass, such as
128 requested tokens, before making TG performance claims.

The 256K request was accepted and entered generation. It remained stable in
unified memory but did not complete within the 30-minute client bound. No swap,
OOM, or context-length rejection occurred.

Raw JSON samples are written by `Scripts/context_ladder.py` to the local,
ignored `benchmark-results/context-ladder/` directory.
