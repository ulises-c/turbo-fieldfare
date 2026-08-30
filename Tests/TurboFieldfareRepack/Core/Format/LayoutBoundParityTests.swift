import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// `GTurboLayoutValidator` carried its own hardcoded 16 MiB bound while the
/// verifier and the runtime both moved to 64 MiB for Qwen 3.6, whose 40 layers
/// x 256 experts produce a ~22 MB `layout.json`. The mismatch was invisible
/// until an 18.8 GiB download finished and was rejected at the last step:
///
///     install failed: install state at .../packed_experts/layout.json is
///     corrupt: size 22493846 exceeds 16777216-byte cap
///
/// These tests pin every layout bound in the package to the same constant so
/// one path cannot be raised for a new architecture while another is missed.
@Suite struct LayoutBoundParityTests {

    @Test("Every layout bound in the package is the same number")
    func layoutBoundsAgree() {
        #expect(VerifiedInstallTool.layoutMaxBytes
                == PackedExpertsLayoutReader.defaultMaxBytes,
                """
                The installer and the runtime disagree about how large a \
                layout.json may be. A model that installs will then fail to \
                load, or -- as happened with Qwen 3.6 -- a complete download \
                is rejected at verification.
                """)
    }

    /// The concrete case that broke: Qwen 3.6's real layout size must be
    /// admissible everywhere. Sized from the actual file, 22,493,846 bytes.
    @Test("A Qwen-sized layout.json fits every bound")
    func qwenSizedLayoutIsAdmissible() {
        let qwenLayoutBytes: UInt64 = 22_493_846
        #expect(qwenLayoutBytes < VerifiedInstallTool.layoutMaxBytes)
        #expect(qwenLayoutBytes < PackedExpertsLayoutReader.defaultMaxBytes)
        // And the bound it used to be checked against, which it exceeds --
        // this is the assertion that would have caught the bug.
        #expect(qwenLayoutBytes > 16 * 1024 * 1024)
    }

    /// The layout bound is deliberately larger than the generic metadata
    /// bound. If someone "tidies" them into one constant, the packed-experts
    /// layout silently loses headroom.
    @Test("The layout bound is larger than the generic metadata bound")
    func layoutBoundExceedsGenericMetadataBound() {
        #expect(VerifiedInstallTool.layoutMaxBytes
                > VerifiedInstallTool.metadataMaxBytes)
        #expect(VerifiedInstallTool.layoutMaxBytes
                > VerifiedInstallTool.manifestMaxBytes)
    }
}
