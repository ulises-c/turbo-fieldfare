import Foundation

/// Opt-in GPU command-buffer timing used by the profiling CLI.
public enum KernelGPUStats {
    public static let environmentKey = "TURBO_FIELDFARE_KERNEL_STATS"

    /// Matches the repository's opt-in environment flag convention exactly.
    public static func isEnabled(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment[environmentKey] == "1"
    }
}

public struct KernelGPUTimingSummary: Equatable, Sendable {
    public let role: String
    public let gpuMilliseconds: Double
    public let count: Int

    public init(role: String, gpuMilliseconds: Double, count: Int) {
        self.role = role
        self.gpuMilliseconds = gpuMilliseconds
        self.count = count
    }
}

/// Aggregates by role as samples arrive, so storage is bounded by role count
/// rather than token count or runner lifetime.
struct KernelGPUTimingAccumulator {
    private struct Aggregate {
        var gpuMilliseconds = 0.0
        var count = 0
    }

    private var byRole: [String: Aggregate] = [:]

    mutating func record(role: String, start: TimeInterval, end: TimeInterval) {
        guard end > start else { return }
        var aggregate = byRole[role, default: Aggregate()]
        aggregate.gpuMilliseconds += (end - start) * 1_000
        aggregate.count += 1
        byRole[role] = aggregate
    }

    mutating func reset() {
        byRole.removeAll(keepingCapacity: true)
    }

    func summary() -> [KernelGPUTimingSummary] {
        byRole.map {
            KernelGPUTimingSummary(role: $0.key,
                                   gpuMilliseconds: $0.value.gpuMilliseconds,
                                   count: $0.value.count)
        }
        .sorted { $0.role < $1.role }
    }
}
