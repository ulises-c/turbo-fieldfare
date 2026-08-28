import Foundation

/// The single admission rule for `--max-context` and prefill, shared by the
/// CLI, the server, and the Mac app.
///
/// Before this existed each surface decided for itself. The server enforced a
/// KV budget, the app offered a fixed menu, and the CLI accepted any positive
/// integer — so `--max-context 1000000` was rejected by one entry point and
/// silently accepted by another, failing later inside the allocator. The
/// difference was invisible until a run died deep in a prefill chunk.
///
/// Every bound here is derived from the loaded `ArchConfig`, never from a
/// constant that happens to match one model.
public enum ContextAdmission {
    /// The largest context any supported model advertises. Both Gemma 4
    /// 26B-A4B and Qwen 3.6 35B-A3B are natively 262,144.
    public static let nativeMaximumContext = 262_144

    /// KV budget above which unchunked prefill is refused. This is a memory
    /// bound rather than a context bound on purpose: the two families differ
    /// by 11x in unringed KV at the same context, so a single context limit
    /// would either admit a Gemma configuration that needs 55 GiB or reject a
    /// Qwen one that needs 5 GiB.
    public static let maximumUnchunkedKVBytes = 16 * 1_073_741_824

    public enum Rejection: Error, Equatable, CustomStringConvertible {
        case notPositive(maxContext: Int)
        case aboveNativeMaximum(maxContext: Int, native: Int, family: String)
        case unchunkedKVTooLarge(maxContext: Int, family: String, gibibytes: Int)

        /// Renders the rejection against the surface's own vocabulary.
        ///
        /// `subject` is the complete leading noun phrase naming the offending
        /// context — the server passes `"--max-context 262144"`, while the CLI
        /// passes just `"262144"` because its error wrapper has already printed
        /// the flag. The number is never appended separately, so no caller can
        /// end up printing it twice.
        public func message(subject: String, prefillRemedy: String) -> String {
            switch self {
            case .notPositive:
                return "\(subject) must be positive"
            case .aboveNativeMaximum(_, let native, let family):
                return "\(subject) is above the native maximum \(native) for "
                    + "\(family); the model has no trained positions beyond that"
            case .unchunkedKVTooLarge(_, let family, let gibibytes):
                return "\(subject) requires \(prefillRemedy) for \(family); "
                    + "without chunked prefill every layer allocates KV at the "
                    + "full context instead of the sliding-window ring, which "
                    + "needs about \(gibibytes) GiB of KV alone"
            }
        }

        public var description: String {
            switch self {
            case .notPositive(let maxContext),
                 .aboveNativeMaximum(let maxContext, _, _),
                 .unchunkedKVTooLarge(let maxContext, _, _):
                return message(subject: "max context \(maxContext)",
                               prefillRemedy: "chunked prefill")
            }
        }
    }

    /// Checks a context and prefill choice against one architecture.
    ///
    /// Pass `family: nil` when the model has not been identified yet, as at
    /// argument-parse time. The family-independent bounds still apply, but the
    /// KV budget is evaluated against the most permissive known architecture so
    /// parsing never rejects a configuration that the real model would accept.
    /// The strict check runs later, once the manifest has been read.
    public static func check(maxContext: Int,
                             family: ModelFamily?,
                             prefillEnabled: Bool,
                             prefillChunkTokens: Int) throws {
        guard maxContext > 0 else {
            throw Rejection.notPositive(maxContext: maxContext)
        }
        guard maxContext <= nativeMaximumContext else {
            throw Rejection.aboveNativeMaximum(
                maxContext: maxContext,
                native: nativeMaximumContext,
                family: family?.rawValue ?? "every supported model")
        }
        guard !prefillEnabled else { return }
        let candidates: [ModelFamily] = family.map { [$0] }
            ?? Array(ArchConfig.knownArchitectures.keys)
        // With no family named, admit if ANY known architecture could run it.
        var smallest: (family: ModelFamily, bytes: Int)?
        for candidate in candidates {
            let arch = ArchConfig.knownArchitectures[candidate] ?? .gemma4_26B_A4B
            let bytes = arch.kvFootprint(maxContext: maxContext,
                                         prefillChunkTokens: prefillChunkTokens,
                                         ringEnabled: false).totalBytes
            if smallest == nil || bytes < smallest!.bytes {
                smallest = (candidate, bytes)
            }
        }
        guard let best = smallest, best.bytes > maximumUnchunkedKVBytes else {
            return
        }
        let gib = Int((Double(best.bytes) / 1_073_741_824).rounded())
        throw Rejection.unchunkedKVTooLarge(maxContext: maxContext,
                                            family: best.family.rawValue,
                                            gibibytes: gib)
    }

    /// The context ladder the app menu and the docs both present. Every rung
    /// is a power-of-two multiple or a 32K step, ending at the native maximum.
    public static let ladder = [4_096, 8_192, 16_384, 32_768, 65_536,
                                98_304, 131_072, 196_608, 262_144]
}
