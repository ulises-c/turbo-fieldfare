import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    /// The menu is the shared ladder, not a parallel list. Deriving the
    /// expectation from `ContextAdmission.ladder` means adding a rung to one
    /// and not the other fails here instead of shipping a menu entry the CLI
    /// and server would reject.
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == ContextAdmission.ladder)
    }

    /// The top option is the native context both supported families advertise.
    @Test func theLargestOptionIsTheNativeMaximum() {
        #expect(AppContextLengthOption.allCases.last?.tokens
            == ContextAdmission.nativeMaximumContext)
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        // 29 MiB above the older figures at every size, because the sliding
        // ring is sized for the widest prefill chunk the runtime may see - the
        // pooled image-token count - and the estimate had used the smaller text
        // chunk. The menu deltas are unchanged: every option grew equally.
        #expect(mebibytes == [334, 414, 574, 894, 1_534,
                              2_174, 2_814, 4_094, 5_374])
        // Deltas are relative to the default, which is 8K as of 2026-08-17 so
        // that an image and its prompt fit without the user changing anything.
        // Now computed from the shared KV model rather than hand-written, so
        // the figures are exact where the old literals were rounded by hand
        // (-85 -> -84 MB, +170 -> +168 MB, +500 -> +503 MB); +1.17 GB is
        // unchanged, which is what confirms the decimal-unit convention.
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, -84 MB",
            "8K, Default",
            "16K, +168 MB",
            "32K, +503 MB",
            "64K, +1.17 GB",
            "96K, +1.85 GB",
            "128K, +2.52 GB",
            "192K, +3.86 GB",
            "256K, +5.20 GB",
        ])
    }

    /// The estimate must follow the installed architecture. Qwen 3.6 holds a
    /// fixed recurrent state in place of 30 layers of per-token K/V rows, so
    /// its long-context cost is materially lower than Gemma's - reporting a
    /// Gemma-shaped number for a Qwen install would overstate it.
    @Test func longContextEstimateIsArchitectureSpecific() {
        let option = AppContextLengthOption.twoFiftySixK
        let gemma = option.fp16KVBytes(for: .gemma4_26B_A4B)
        let qwen = option.fp16KVBytes(for: .qwen36_35B_A3B)
        #expect(qwen < gemma)
        #expect(gemma == option.fp16KVBytes)
    }
}
