import Foundation

/// Analytic FP16 KV-cache footprint, derived from an `ArchConfig` rather than
/// hardcoded per family.
///
/// Three call sites previously each carried their own copy of the Gemma 4
/// formula (`ServerArguments.unchunkedKVGibibytes`,
/// `AppContextLengthOption.fp16KVBytes`, and the long-context experiment
/// notes). Each of them silently produced a Gemma number for any model, which
/// is wrong for Qwen 3.6: its 30 gated-DeltaNet layers hold a fixed-size
/// recurrent state instead of per-token K/V rows, so a Gemma-shaped estimate
/// overstates its KV by roughly an order of magnitude at long context.
///
/// The arithmetic here mirrors `KVCacheManager.init` exactly — same strides,
/// same ring capacity, same treatment of linear layers — so a change to the
/// allocator that is not reflected here shows up as a failing cross-check
/// test rather than as a quietly wrong estimate.
public struct KVFootprint: Sendable, Equatable {
    /// Per-token K/V rows for full-attention and sliding-window layers.
    public let kvCacheBytes: Int
    /// Fixed-size gated-DeltaNet recurrent state plus depthwise conv tails.
    /// Zero for architectures with no linear-attention layers.
    public let linearStateBytes: Int

    public var totalBytes: Int { kvCacheBytes + linearStateBytes }

    public var totalMebibytes: Double { Double(totalBytes) / 1_048_576 }
    public var totalGibibytes: Double { Double(totalBytes) / 1_073_741_824 }
}

extension ArchConfig {
    private static let fp16Size = 2
    private static let fp32Size = 4
    private static let keyAndValue = 2

    /// Bytes per token held by one sliding-window layer (K and V together).
    public var slidingBytesPerToken: Int {
        numKVHeads * headDim * Self.keyAndValue * Self.fp16Size
    }

    /// Bytes per token held by one full-attention layer (K and V together).
    public var fullBytesPerToken: Int {
        numFullKVHeads * fullHeadDim * Self.keyAndValue * Self.fp16Size
    }

    /// Fixed recurrent state carried by the gated-DeltaNet layers, matching
    /// `GDNStateManager.init`. Independent of context length — this is the
    /// property that makes Qwen 3.6 cheap at long context.
    public var linearAttentionStateBytes: Int {
        guard hasLinearAttentionLayers else { return 0 }
        let linearLayers = fullAttentionLayerMask.filter { $0 == 2 }.count
        let state = linearAttention.numVHeads
            * linearAttention.valueHeadDim
            * linearAttention.keyHeadDim
            * Self.fp32Size
        let convTail = max(0, linearAttention.convKernelSize - 1)
            * linearAttention.qkvDim
            * Self.fp16Size
        return linearLayers * (state + convTail)
    }

    /// Physical rows a sliding-window layer allocates. With the FP16 ring
    /// enabled (which chunked prefill turns on) the window is capped at
    /// `slidingWindow + chunkTokens`; without it every layer allocates the
    /// full context. Mirrors `KVCacheManager`'s `swaCapacity`.
    public func slidingWindowRows(maxContext: Int,
                                  prefillChunkTokens: Int,
                                  ringEnabled: Bool) -> Int {
        guard ringEnabled else { return maxContext }
        return min(maxContext, max(1, slidingWindow + prefillChunkTokens))
    }

    /// Resident FP16 KV footprint at `maxContext`.
    ///
    /// - Parameter ringEnabled: whether the sliding-window ring is active.
    ///   Chunked prefill (`--prefill on`) enables it; `--prefill off` does not.
    public func kvFootprint(maxContext: Int,
                            prefillChunkTokens: Int = 128,
                            ringEnabled: Bool = true) -> KVFootprint {
        precondition(maxContext > 0, "maxContext must be positive")
        let rows = slidingWindowRows(maxContext: maxContext,
                                     prefillChunkTokens: prefillChunkTokens,
                                     ringEnabled: ringEnabled)
        var bytes = 0
        for layer in 0..<numLayers {
            switch fullAttentionLayerMask[layer] {
            case 2:
                // Linear-attention layer: no per-token K/V rows at all. Its
                // cost is the fixed state, accounted separately.
                continue
            case 0:
                bytes += rows * slidingBytesPerToken
            default:
                bytes += maxContext * fullBytesPerToken
            }
        }
        return KVFootprint(kvCacheBytes: bytes,
                           linearStateBytes: linearAttentionStateBytes)
    }

    /// Marginal KV cost of one additional context token, in bytes. Only the
    /// full-attention layers scale with context once the ring is enabled, so
    /// this is the slope of the memory-versus-context line a benchmark plots.
    public var marginalKVBytesPerToken: Int {
        let fullLayers = fullAttentionLayerMask.filter { $0 == 1 }.count
        return fullLayers * fullBytesPerToken
    }
}
