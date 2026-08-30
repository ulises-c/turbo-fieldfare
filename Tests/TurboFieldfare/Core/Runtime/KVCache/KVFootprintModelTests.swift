import Testing
import Metal
@testable import TurboFieldfare

/// The analytic KV model exists so the server guard, the app's context menu,
/// and the benchmark tables all predict the same number. That is only safe if
/// it tracks `KVCacheManager`'s real allocation, so these tests compare the
/// prediction against the allocator itself rather than against a constant.
@Suite struct KVFootprintModelTests {

    // MARK: - Cross-check against the real allocator

    /// The prediction must equal what `KVCacheManager` actually allocates.
    /// A change to the allocator that is not mirrored in the model fails here
    /// instead of silently producing wrong memory estimates.
    @Test(arguments: [4_096, 16_384, 65_536])
    func predictionMatchesTheAllocatorForGemma(_ maxContext: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let arch = ArchConfig.gemma4_26B_A4B
        let chunk = 128
        let cache = try KVCacheManager(device: device,
                                       config: arch,
                                       maxContext: maxContext,
                                       fp16RingEnabled: true,
                                       maxPrefillChunkTokens: chunk)
        let predicted = arch.kvFootprint(maxContext: maxContext,
                                         prefillChunkTokens: chunk,
                                         ringEnabled: true)
        #expect(predicted.kvCacheBytes == cache.allocatedKVBytes)
    }

    @Test(arguments: [4_096, 16_384, 65_536])
    func predictionMatchesTheAllocatorForQwen(_ maxContext: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let arch = ArchConfig.qwen36_35B_A3B
        let chunk = 128
        let cache = try KVCacheManager(device: device,
                                       config: arch,
                                       maxContext: maxContext,
                                       fp16RingEnabled: true,
                                       maxPrefillChunkTokens: chunk)
        let predicted = arch.kvFootprint(maxContext: maxContext,
                                         prefillChunkTokens: chunk,
                                         ringEnabled: true)
        #expect(predicted.kvCacheBytes == cache.allocatedKVBytes)
    }

    // MARK: - Architecture-dependent behavior

    /// Qwen's linear-attention layers hold no per-token K/V rows, so the ring
    /// changes nothing for it. This is the property that makes a single
    /// context-based `--prefill off` bound wrong.
    @Test func ringIsANoOpForAnArchitectureWithoutSlidingWindowLayers() {
        let arch = ArchConfig.qwen36_35B_A3B
        let ringed = arch.kvFootprint(maxContext: 262_144, ringEnabled: true)
        let unringed = arch.kvFootprint(maxContext: 262_144, ringEnabled: false)
        #expect(ringed.totalBytes == unringed.totalBytes)
    }

    /// Matched control: the same call on Gemma must differ sharply, otherwise
    /// the test above is passing for the wrong reason.
    @Test func ringMattersForAnArchitectureWithSlidingWindowLayers() {
        let arch = ArchConfig.gemma4_26B_A4B
        let ringed = arch.kvFootprint(maxContext: 262_144, ringEnabled: true)
        let unringed = arch.kvFootprint(maxContext: 262_144, ringEnabled: false)
        #expect(unringed.totalBytes > ringed.totalBytes * 8)
    }

    /// Gemma has no linear layers and therefore no recurrent state; Qwen's is
    /// fixed and does not grow with context.
    @Test func linearStateIsFixedAndGemmaHasNone() {
        #expect(ArchConfig.gemma4_26B_A4B.linearAttentionStateBytes == 0)
        let qwen = ArchConfig.qwen36_35B_A3B
        #expect(qwen.linearAttentionStateBytes > 0)
        let small = qwen.kvFootprint(maxContext: 4_096)
        let large = qwen.kvFootprint(maxContext: 262_144)
        #expect(small.linearStateBytes == large.linearStateBytes)
    }

    /// Only full-attention layers scale with context once the ring is on, so
    /// growth is linear with a known slope. This is the slope the benchmark
    /// tables report.
    @Test(arguments: [ArchConfig.gemma4_26B_A4B, ArchConfig.qwen36_35B_A3B])
    func footprintGrowsLinearlyAtTheAdvertisedSlope(_ arch: ArchConfig) {
        let low = arch.kvFootprint(maxContext: 131_072)
        let high = arch.kvFootprint(maxContext: 262_144)
        let delta = high.totalBytes - low.totalBytes
        #expect(delta == arch.marginalKVBytesPerToken * (262_144 - 131_072))
    }

    /// Both supported families are natively 262,144, and both must stay within
    /// a sane resident budget at that context with the ring enabled.
    @Test(arguments: [ArchConfig.gemma4_26B_A4B, ArchConfig.qwen36_35B_A3B])
    func nativeMaximumContextFitsInAReasonableBudget(_ arch: ArchConfig) {
        let footprint = arch.kvFootprint(maxContext: 262_144)
        #expect(footprint.totalGibibytes < 8.0)
    }
}
