import Testing
@testable import TurboFieldfareServerCore

/// A cancelled request used to be excluded from logging entirely, on the
/// reasoning that a client hanging up is not a server fault. That is true, and it
/// left the request's last line as `generating` forever: reading the log, a
/// running request, an abandoned one and a crashed one were indistinguishable.
@Suite struct ServerLogTests {
    @Test func completionMessageIncludesPromptAndGenerationTimings() {
        let completion = ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 120_013,
                               completionTokens: 1,
                               totalTokens: 120_014,
                               cachedTokens: 0),
            prefillSeconds: 675.595,
            decodeSeconds: 0.250)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-timing",
            duration: .seconds(676),
            completion: completion)

        #expect(line.contains("pp=675.595s"))
        #expect(line.contains("tg=0.250s"))
        #expect(line.contains("tg_tok_s=4.000"))
    }

    @Test func aCancelledRequestReadsAsCancelledRatherThanFailed() {
        let line = ServerLog.cancelledMessage(id: "chatcmpl-abc",
                                              phase: "generating",
                                              duration: .milliseconds(5369))

        #expect(line.contains("chatcmpl-abc"))
        #expect(line.contains("cancelled by client"))
        #expect(line.contains("phase=generating"))
        #expect(line.contains("5.369s"))
        // Distinct from a failure, so an abandoned client does not read as a
        // server fault in the log or in anything counting error lines.
        #expect(!line.contains("failed"))
    }
}
