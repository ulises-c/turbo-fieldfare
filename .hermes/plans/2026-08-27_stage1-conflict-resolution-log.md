# Stage 1 conflict-resolution log — feat/qwen36-arch

Merge of `pr29` (upstream PR #29, Qwen 3.6 35B-A3B) onto fork-main synced to upstream/main @ #159.
20 conflicted files, ~1,090 conflicted lines. Resolved by hand; this records WHY, since a
`-X ours` / `-X theirs` shortcut would have silently dropped behavior in several places.

## Root cause of the conflicts

PR29 branched from a stale upstream base (`add22ff`, PR #49) and predates:

- `fd3c0b8` optional Gemma 4 image support (#144) — touches 5 of the 6 hotspots
- `ec8ad21` prefill speedup on pre-Apple10 Macs (#159)
- `417f389` CLI runtime controls / `RuntimeConfiguration` (#100)
- `acefaf1` incremental lossless detokenization (#118)
- `d56c808` typed Codable manifest contract (#87)

Both PR29 and upstream generalize the SAME seams in different directions: upstream adds a
MODALITY (text|image), PR29 adds a FAMILY (gemma4|qwen36). The resolution treats these as two
independent axes rather than one flat enum.

## Governing rule

Where PR29 and upstream disagree about PRE-EXISTING machinery, upstream wins (PR29 is stale).
PR29's genuinely NEW additions are preserved. Anything else is a regression.

## Non-obvious decisions

### Format / manifest
- `GTurboManifestV1`: extended the TYPED Codable contract with 13 optional family fields rather
  than accepting PR29's hand-rolled `[String: Any]` dict. `JSONEncoder` omits nil optionals and
  `.sortedKeys` fixes ordering, so **Gemma manifests still encode byte-identically**. PR29's own
  test asserts Gemma omits these keys — satisfied.
- Factored the per-field `family == .gemma4 ? nil : value` repetition into one
  `writesFamilyExtensions` flag, so adding Ornith later is a one-line change, not 13 ternaries.
- `VerifiedInstallTool`: upstream split one constant into metadata/manifest/layout caps; PR29
  raised the single old one. Kept the split, raised ONLY `layoutMaxBytes` (Qwen's layout.json is
  ~22 MB: 40 layers x 256 experts). Raising the manifest cap too would have been a security
  regression with no benefit.

### Runtime (RealForwardRunner — the hard one, 3 hunks / 365 lines)
- PR29 wraps prefill attention in `if isLinear {...} else {...}`; upstream added the FP16-KV guard
  and bidirectional-block params INSIDE that same region. `-X ours`/`-X theirs` drops one of them.
  Resolution: PR29's outer family dispatch, with upstream's `layerKind:`,
  `bidirectionalBlockStart/End`, and `fp16KV:` restored in the else branch.
  **PR29's call had silently dropped these — that would have been a Gemma image regression.**
- Hunk 3 was a pure insertion (two Qwen decode helpers); kept upstream's `throws` signature on
  `runSync` since its body calls `try waitForCompletion`.

### Tokenization
- The auto-merge mangled `resolveGemmaTokens`: upstream's inline `self.x =` assignments ended up
  inside PR29's struct-returning function. Rewrote the region so upstream's hardened
  `requireTokenID` (which throws `specialTokenMismatch`) returns PR29's struct shape.
- `StructuredAssistantDecoder`: dialect dispatch FIRST, then upstream's Gemma control-token
  routing — composition, not replacement.

### Server
- `OpenAIModels.validate` now takes BOTH `dialect:` and upstream's `preStagedImages:` /
  `attachmentLease:`. This is the family x modality seam.
- `GemmaToolSchema.adapted()` and `gemmaToolArgumentBody()` are Gemma-specific prompt encodings;
  they are now skipped for ChatML, which emits `<tool_call>` JSON directly. Applying Gemma's
  adapter to Qwen would corrupt tool schemas.

### CLI
- `Args.swift` / `Run.swift`: took upstream wholesale (PR29 predates `RuntimeConfiguration`,
  `--prefill*`, `--image`, `--vision-*`), then re-added ONLY PR29's genuine addition:
  `RawCompletionScratch(..., logitSoftcap: Float(model.config.finalLogitSoftcap))`.
  **Without this, Qwen would be silently softcapped at Gemma's default 30.0 — wrong logits.**

### Fail-closed image path (AGENTS.md requirement)
- `AppModelInstallDescriptor.visionCompanion(for:)` returns nil for `.qwen36`. Callers must treat
  nil as "image support unavailable" instead of falling back to Gemma's pack, which would pair one
  family's vision tower with another family's text model.
- `TurboFieldfareRepack`: `--model` is now REJECTED when combined with vision install flags
  (previously silently ignored). The image pack pins `SupportedModelSource.gemma4` explicitly.
- Runtime/server-side rejection of image input for Qwen is still TODO (task 10).

## Post-resolution compile fixes
- `ManifestReader.peekFamily` called `metadataFileSize`, a helper upstream refactored away.
  Rewritten onto upstream's hardened `GTurboModelDirectory.readMetadata(_:maxBytes:)` so family
  detection cannot read anything `load` would refuse.
- `ManifestArch(wire:)` threaded the 13 new family fields.
- Repack `main.swift` used `SupportedModelSource.repoID` statically -> `.gemma4.repoID`.

## Gates before this merge is committed
- Gate A: `swift build -c release` green + `Scripts/test.sh` (delegated).
- Gate B: Gemma 4 byte-identical output vs fork-main at fixed seeds. Anything less is a blocking defect.

## Stage 2 (not started)
Cherry-pick NVMAI perf commits individually behind opt-in flags. Pick `927c94e` FIRST: it fixes a
profiler bug that reported routed MoE at 0.33 ms/token against a true 14.4 (~40x undersample).
Every later NVMAI attribution was measured on that broken instrument. Their figures are all
M3 24 GB; on this M5 Max they are hypotheses, not results.
Two picks flip runtime DEFAULTS (64 expert slots + mlock, 4096-token prefill chunks) — AGENTS.md
forbids that unprompted, so they land as flags and a default-flip is proposed only with our own numbers.
