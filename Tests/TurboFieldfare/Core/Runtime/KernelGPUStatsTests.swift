import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct KernelGPUStatsTests {
    @Test func flagEnablesOnlyForExactOne() {
        #expect(KernelGPUStats.isEnabled(environment: [:]) == false)
        #expect(KernelGPUStats.isEnabled(environment: ["TURBO_FIELDFARE_KERNEL_STATS": ""]) == false)
        #expect(KernelGPUStats.isEnabled(environment: ["TURBO_FIELDFARE_KERNEL_STATS": "0"]) == false)
        #expect(KernelGPUStats.isEnabled(environment: ["TURBO_FIELDFARE_KERNEL_STATS": "false"]) == false)
        #expect(KernelGPUStats.isEnabled(environment: ["TURBO_FIELDFARE_KERNEL_STATS": "true"]) == false)
        #expect(KernelGPUStats.isEnabled(environment: ["TURBO_FIELDFARE_KERNEL_STATS": "1"]) == true)
    }

    @Test func accumulatorResetIsolatesGenerations() {
        var accumulator = KernelGPUTimingAccumulator()
        accumulator.record(role: "embed", start: 1.0, end: 1.002)
        #expect(accumulator.summary().count == 1)

        accumulator.reset()

        #expect(accumulator.summary().isEmpty)
        accumulator.record(role: "fused_head", start: 2.0, end: 2.003)
        let summary = accumulator.summary()
        #expect(summary.count == 1)
        #expect(summary.first?.role == "fused_head")
        #expect(summary.first?.count == 1)
        #expect(abs((summary.first?.gpuMilliseconds ?? 0) - 3.0) < 0.000_001)
    }

    @Test func cliReportIsNonemptyAndDeterministic() {
        let report = kernelGPUStatsReport(summary: [
            KernelGPUTimingSummary(role: "embed", gpuMilliseconds: 2, count: 1),
            KernelGPUTimingSummary(role: "attention_router", gpuMilliseconds: 4, count: 2),
        ], tokenCount: 2)

        #expect(report == """
        [kernel role=attention_router gpu_ms=4.000 per_token_ms=2.000 count=2]
        [kernel role=embed gpu_ms=2.000 per_token_ms=1.000 count=1]
        [kernel role=total gpu_ms=6.000 per_token_ms=3.000 count=3]

        """)
    }

    @Test func cliReportConsumerUsesExactFlagGate() {
        let summary = [
            KernelGPUTimingSummary(role: "embed", gpuMilliseconds: 2, count: 1),
        ]
        #expect(kernelGPUStatsReportIfEnabled(
            summary: summary,
            tokenCount: 1,
            environment: [KernelGPUStats.environmentKey: "0"]).isEmpty)
        #expect(kernelGPUStatsReportIfEnabled(
            summary: summary,
            tokenCount: 1,
            environment: [KernelGPUStats.environmentKey: "1"])
            .contains("[kernel role=total"))
    }

    @Test func cliReportUsesCapturedDecodeStepsForPerTokenTiming() {
        let report = kernelGPUStatsReport(summary: [
            KernelGPUTimingSummary(
                role: "attention_router", gpuMilliseconds: 630, count: 1_890),
            KernelGPUTimingSummary(role: "logits_head", gpuMilliseconds: 63, count: 63),
        ], tokenCount: 64)

        #expect(report.contains(
            "[kernel role=attention_router gpu_ms=630.000 per_token_ms=10.000 count=1890]"))
        #expect(report.contains(
            "[kernel role=logits_head gpu_ms=63.000 per_token_ms=1.000 count=63]"))
    }
}
