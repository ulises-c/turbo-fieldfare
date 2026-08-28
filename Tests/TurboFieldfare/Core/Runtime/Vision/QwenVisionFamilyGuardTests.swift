import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// The image tower is Gemma 4's. `VisionRuntime.open` is the single door into
/// it, and it opens the *text* model's manifest expecting
/// `ArchConfig.gemma4_26B_A4B`, so a Qwen 3.6 text model can never be paired
/// with Gemma's vision pack.
///
/// These cases pin that guard to the `family` field specifically. Asserting
/// only "it threw" would keep passing if the family check were dropped and the
/// throw came from some incidental shape mismatch instead — and it would keep
/// passing if a future Qwen text model happened to share Gemma's dimensions.
/// Nothing here needs an installed model or a vision pack: the family check
/// runs on `manifest.json` alone, before any pack is looked for.
@Suite("Qwen text models are refused by the vision family guard")
struct QwenVisionFamilyGuardTests {

    // MARK: - Synthetic manifests (no model, no weights, no downloads)

    /// A structurally valid `manifest.json` describing `arch`, written into a
    /// fresh temporary `.gturbo` directory. Only the manifest is written: the
    /// family check fires inside `ManifestReader.load` before any weight file,
    /// expert layout, or vision companion is touched.
    private static func writeManifestOnlyModel(
        arch: ArchConfig,
        familyOverride: String? = nil,
        modelID: String = "synthetic-family-guard"
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "family-guard-\(UUID().uuidString).gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        let zeroSHA = String(repeating: "0", count: 64)
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) },
            "family": familyOverride ?? arch.family.rawValue,
            "attnOutputGate": arch.attnOutputGate,
            "attentionScale": arch.attentionScale,
            "embeddingScaledBySqrtHidden": arch.embeddingScaledBySqrtHidden,
            "routerScaled": arch.routerScaled,
            "ffnSandwichNorms": arch.ffnSandwichNorms,
            "sharedExpertGated": arch.sharedExpertGated,
            "ropeNeoxSubdim": arch.ropeNeoxSubdim,
        ]
        let linear = arch.linearAttention
        if linear != .none {
            archDict["linearNumKHeads"] = linear.numKHeads
            archDict["linearNumVHeads"] = linear.numVHeads
            archDict["linearKeyHeadDim"] = linear.keyHeadDim
            archDict["linearValueHeadDim"] = linear.valueHeadDim
            archDict["linearConvKernelSize"] = linear.convKernelSize
        }

        func quantSlot(_ weightBits: Int) -> [String: Any] {
            [
                "weightBits": weightBits,
                "scheme": "affine",
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": Quantization.groupSize,
            ]
        }

        let root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": modelID,
            "sourceSnapshotHash": "synthetic-snapshot",
            "arch": archDict,
            "files": [
                "model_weights.bin": ["size": 16_384, "sha256": zeroSHA],
                "packed_experts/layout.json": ["size": 64, "sha256": zeroSHA],
            ],
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": GTurboFormatV1.alignmentBytes,
            "quant": [
                "embedding": quantSlot(4),
                "attention": quantSlot(4),
                "router": quantSlot(8),
                "sharedExpert": quantSlot(4),
                "routedExpert": quantSlot(4),
            ],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes])
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// The `field` of a `ModelError.archMismatch`, or nil for anything else.
    /// Used so a case can say *which* guard rejected the model rather than
    /// settling for "some error was thrown".
    private static func archMismatchField(_ error: Error) -> String? {
        guard case ModelError.archMismatch(let field, _, _) = error else { return nil }
        return field
    }

    // MARK: - The guard itself, on any hardware

    /// The exact load `VisionRuntime.open` performs. A manifest identical to
    /// Gemma 4 in every arch field but declaring `family: "qwen36"` must still
    /// be rejected, and rejected *on the family field* — that is the only
    /// assertion that fails if the family check is deleted or widened, since
    /// every other field matches.
    @Test func gemmaShapedQwenManifestIsRejectedOnTheFamilyField() throws {
        let directory = try Self.writeManifestOnlyModel(
            arch: .gemma4_26B_A4B, familyOverride: ModelFamily.qwen36.rawValue)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect {
            _ = try ManifestReader.load(directoryURL: directory,
                                        expecting: .gemma4_26B_A4B)
        } throws: { error in
            guard case ModelError.archMismatch(
                let field, let expected, let actual) = error else { return false }
            return field == "family"
                && expected == ModelFamily.gemma4.rawValue
                && actual == ModelFamily.qwen36.rawValue
        }
    }

    /// The same manifest with the family put back loads, so the case above is
    /// isolating the family field rather than tripping over a broken fixture.
    @Test func theSameManifestLoadsOnceTheFamilyIsGemma() throws {
        let directory = try Self.writeManifestOnlyModel(arch: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifest = try ManifestReader.load(directoryURL: directory,
                                               expecting: .gemma4_26B_A4B)
        #expect(manifest.arch.family == ModelFamily.gemma4.rawValue)
    }

    /// A real-shaped Qwen 3.6 35B-A3B text model — the family this branch
    /// added — is refused by the same load, as an arch mismatch and not as a
    /// missing file, a corrupt index, or anything else the vision path might
    /// later paper over.
    @Test func productionQwen36ManifestIsRejectedAsAnArchMismatch() throws {
        let directory = try Self.writeManifestOnlyModel(arch: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: directory) }

        var thrown: Error?
        do {
            _ = try ManifestReader.load(directoryURL: directory,
                                        expecting: .gemma4_26B_A4B)
        } catch {
            thrown = error
        }
        let error = try #require(thrown, "a Qwen 3.6 text model loaded as Gemma 4")
        let field = try #require(
            Self.archMismatchField(error),
            "expected ModelError.archMismatch, got \(String(reflecting: error))")
        // Qwen 3.6 differs from Gemma 4 in many dimensions; whichever is
        // checked first, the rejection must be an arch mismatch.
        #expect(!field.isEmpty)
    }

    // MARK: - Through the single vision entry point

    /// `VisionRuntime.open` is the only door to the image tower. Handed a Qwen
    /// 3.6 text model it must throw rather than return a runtime — and throw
    /// the family mismatch, not a "vision pack not found" that would let a
    /// caller conclude the pairing itself was fine.
    ///
    /// No vision pack is created beside the model on purpose: if the family
    /// guard were ever removed, this would start failing with
    /// `VisionPackError.packNotFound`, which is exactly the silent widening the
    /// case exists to catch.
    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "the image tower requires an M2 or newer Apple Silicon Mac"))
    func visionRuntimeOpenRefusesAQwenTextModelOnTheFamilyField() throws {
        let context = try MetalContext()
        let directory = try Self.writeManifestOnlyModel(
            arch: .gemma4_26B_A4B, familyOverride: ModelFamily.qwen36.rawValue)
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect {
            _ = try VisionRuntime.open(textModelURL: directory, context: context)
        } throws: { error in
            guard case ModelError.archMismatch(
                let field, _, let actual) = error else { return false }
            return field == "family" && actual == ModelFamily.qwen36.rawValue
        }
    }

    /// The production Qwen 3.6 arch through the same door.
    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "the image tower requires an M2 or newer Apple Silicon Mac"))
    func visionRuntimeOpenRefusesTheProductionQwen36Arch() throws {
        let context = try MetalContext()
        let directory = try Self.writeManifestOnlyModel(arch: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: directory) }

        var thrown: Error?
        do {
            _ = try VisionRuntime.open(textModelURL: directory, context: context)
        } catch {
            thrown = error
        }
        let error = try #require(
            thrown, "VisionRuntime.open paired a Qwen 3.6 text model with Gemma's tower")
        #expect(Self.archMismatchField(error) != nil,
                "expected ModelError.archMismatch, got \(String(reflecting: error))")
    }
}
