import TurboFieldfare

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536
    case ninetySixK = 98_304
    case oneTwentyEightK = 131_072
    case oneNinetyTwoK = 196_608
    case twoFiftySixK = 262_144

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    /// Resident FP16 KV bytes for this context under the given architecture.
    ///
    /// Derived from the shared `ArchConfig.kvFootprint` model rather than an
    /// inlined Gemma formula, so a Qwen 3.6 install reports its own (much
    /// smaller) long-context cost instead of a Gemma-shaped estimate.
    ///
    /// The ring is sized with the widest prefill chunk the app may see, which
    /// is the pooled image-token count rather than the default text chunk.
    /// Using the smaller number understates every estimate by about 29.69 MiB.
    public func fp16KVBytes(for architecture: ArchConfig) -> UInt64 {
        let chunkTokens = max(PrefillRuntimeConfig.defaultChunked.chunkTokens,
                              VisionConfig().maximumPooledTokens)
        let footprint = architecture.kvFootprint(maxContext: tokens,
                                                 prefillChunkTokens: chunkTokens,
                                                 ringEnabled: true)
        return UInt64(footprint.totalBytes)
    }

    /// Gemma 4 remains the default for callers that have not yet resolved the
    /// installed family.
    public var fp16KVBytes: UInt64 { fp16KVBytes(for: .gemma4_26B_A4B) }

    /// Menu label with the delta measured against the 8K default, not against
    /// 4K: moving the default without moving the baseline would have left every
    /// delta describing a size the user is no longer starting from.
    ///
    /// Computed rather than hand-written, so the four long-context options
    /// cannot drift from the allocation they describe. Units are decimal
    /// MB/GB, matching the `+1.17 GB` convention the literals already used.
    public func menuLabel(for architecture: ArchConfig) -> String {
        if self == .eightK { return "8K, Default" }
        let baseline = Int64(AppContextLengthOption.eightK.fp16KVBytes(for: architecture))
        let delta = Int64(fp16KVBytes(for: architecture)) - baseline
        let sign = delta < 0 ? "-" : "+"
        let magnitude = Double(abs(delta))
        let formatted = magnitude >= 1_000_000_000
            ? String(format: "%.2f GB", magnitude / 1_000_000_000)
            : String(format: "%.0f MB", magnitude / 1_000_000)
        return "\(shortLabel), \(sign)\(formatted)"
    }

    public var menuLabel: String { menuLabel(for: .gemma4_26B_A4B) }
}
