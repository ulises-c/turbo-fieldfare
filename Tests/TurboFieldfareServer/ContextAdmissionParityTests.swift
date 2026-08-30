import Testing
import Foundation
@testable import TurboFieldfare
@testable import TurboFieldfareCLICore
@testable import TurboFieldfareServerCore

/// The CLI, the server, and the app each decide whether a `--max-context` is
/// legal. They used to decide differently: the server enforced a KV budget,
/// the app offered a fixed menu, and the CLI accepted any positive integer, so
/// `--max-context 1000000` was refused by one entry point and accepted by
/// another before dying inside the allocator.
///
/// These tests pin the surfaces to each other through their real argument
/// parsers. A change to one that is not made to the others fails here rather
/// than in a user's run.
@Suite("Context admission agrees across every surface")
struct ContextAdmissionParityTests {

    private func cliAccepts(maxContext: Int,
                            prefillOn: Bool,
                            family: ModelFamily) -> Bool {
        guard let args = try? Args.parse([
            "--model", "/tmp/model.gturbo",
            "--prompt", "hi",
            "--max-context", "\(maxContext)",
            "--prefill", prefillOn ? "on" : "off",
            "--expert-cache-slots", "16",
        ]) else { return false }
        return (try? args.resolvedRuntimeConfiguration(
            forceLogitsHead: false, family: family)) != nil
    }

    private func serverAccepts(maxContext: Int,
                               prefillOn: Bool,
                               family: ModelFamily) -> Bool {
        guard let args = try? ServerArguments.parse([
            "--model", "/tmp/model.gturbo",
            "--max-context", "\(maxContext)",
            "--prefill", prefillOn ? "on" : "off",
            "--expert-cache-slots", "16",
        ]) else { return false }
        return (try? args.resolvedRuntimeConfiguration(family: family)) != nil
    }

    @Test("CLI and server agree on every ladder rung, for both families and both prefill modes")
    func cliAndServerAgreeOnTheLadder() throws {
        for family in [ModelFamily.gemma4, .qwen36] {
            for context in ContextAdmission.ladder {
                for prefillOn in [true, false] {
                    let cliOK = cliAccepts(maxContext: context,
                                           prefillOn: prefillOn, family: family)
                    let serverOK = serverAccepts(maxContext: context,
                                                 prefillOn: prefillOn, family: family)
                    #expect(cliOK == serverOK,
                            """
                            CLI and server disagree at \(family.rawValue) \
                            context \(context) prefill \(prefillOn): \
                            cli=\(cliOK) server=\(serverOK)
                            """)
                }
            }
        }
    }

    @Test("Every rung the app menu offers is accepted by the CLI and the server")
    func appMenuMatchesTheAdmissibleLadder() throws {
        // The app's menu is the ladder; anything it offers must be loadable
        // with the app's own default of chunked prefill.
        for context in ContextAdmission.ladder {
            for family in [ModelFamily.gemma4, .qwen36] {
                #expect(cliAccepts(maxContext: context, prefillOn: true, family: family),
                        "CLI rejects menu rung \(context) for \(family.rawValue)")
                #expect(serverAccepts(maxContext: context, prefillOn: true, family: family),
                        "server rejects menu rung \(context) for \(family.rawValue)")
            }
        }
    }

    @Test("The CLI refuses a context above the native maximum instead of failing in the allocator")
    func cliRefusesAboveNativeMaximum() throws {
        #expect(!cliAccepts(maxContext: 1_000_000, prefillOn: true, family: .gemma4))
        #expect(!serverAccepts(maxContext: 1_000_000, prefillOn: true, family: .gemma4))
    }

    @Test("The native maximum itself is admissible on both surfaces and families")
    func nativeMaximumIsAdmissible() throws {
        let native = ContextAdmission.nativeMaximumContext
        for family in [ModelFamily.gemma4, .qwen36] {
            #expect(cliAccepts(maxContext: native, prefillOn: true, family: family))
            #expect(serverAccepts(maxContext: native, prefillOn: true, family: family))
        }
    }

    /// The matched control for the family-aware budget: at 256K with prefill
    /// off, Qwen is admissible and Gemma is not. If a future change collapses
    /// the budget back into a single context constant, exactly one of these
    /// flips and the pair catches it.
    @Test("Unchunked prefill at 256K is refused for Gemma and allowed for Qwen")
    func unchunkedPrefillIsFamilyDependent() throws {
        #expect(!cliAccepts(maxContext: 262_144, prefillOn: false, family: .gemma4))
        #expect(!serverAccepts(maxContext: 262_144, prefillOn: false, family: .gemma4))
        #expect(cliAccepts(maxContext: 262_144, prefillOn: false, family: .qwen36))
        #expect(serverAccepts(maxContext: 262_144, prefillOn: false, family: .qwen36))
    }

    @Test("A zero or negative context is refused everywhere")
    func nonPositiveContextIsRefused() throws {
        for bad in [0, -1] {
            #expect(!cliAccepts(maxContext: bad, prefillOn: true, family: .gemma4))
            #expect(!serverAccepts(maxContext: bad, prefillOn: true, family: .gemma4))
        }
    }

    /// Argument parsing runs before the manifest is read, so it cannot know
    /// the family. An unnamed family must therefore admit whatever ANY
    /// supported model could run, or the CLI would reject a Qwen-legal command
    /// before ever looking at the model. The strict, family-specific check runs
    /// afterwards in the run path.
    @Test("An unidentified family admits anything a supported model could run")
    func unknownFamilyIsPermissive() throws {
        try ContextAdmission.check(maxContext: 262_144,
                                   family: nil,
                                   prefillEnabled: false,
                                   prefillChunkTokens: 128)
        // Named as Gemma the very same configuration is refused.
        #expect(throws: ContextAdmission.Rejection.self) {
            try ContextAdmission.check(maxContext: 262_144,
                                       family: .gemma4,
                                       prefillEnabled: false,
                                       prefillChunkTokens: 128)
        }
    }

    /// Bounds that do not depend on the architecture must still apply when the
    /// family is unknown, otherwise parse-time validation would let an
    /// impossible context through to the allocator.
    @Test("Family-independent bounds still apply with no family named")
    func unknownFamilyStillEnforcesUniversalBounds() throws {
        #expect(throws: ContextAdmission.Rejection.self) {
            try ContextAdmission.check(maxContext: 1_000_000, family: nil,
                                       prefillEnabled: true,
                                       prefillChunkTokens: 128)
        }
        #expect(throws: ContextAdmission.Rejection.self) {
            try ContextAdmission.check(maxContext: 0, family: nil,
                                       prefillEnabled: true,
                                       prefillChunkTokens: 128)
        }
    }
}
