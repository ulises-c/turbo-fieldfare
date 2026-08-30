import Foundation
import TurboFieldfareFormat

enum GTurboLayoutValidator {
    static func validate(path: String,
                                plan: RepackPlan,
                                audit: RepackAudit? = nil) throws {
        // Sized by the largest supported architecture, not by Gemma alone.
        // Qwen 3.6 has 40 layers x 256 experts and its layout.json is about
        // 22 MB, so a 16 MiB cap here rejected a fully downloaded, otherwise
        // valid Qwen install at the final verification step.
        // `VerifiedInstallTool.layoutMaxBytes` is the same bound; keep them
        // together.
        let data = try Posix.readBoundedData(
            path, maximumBytes: VerifiedInstallTool.layoutMaxBytes)
        let layout: GTurboPackedExpertsLayoutV1
        do { layout = try GTurboPackedExpertsLayoutCodec.decode(data) }
        catch {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: \(error)")
        }
        var validatedLogicalExperts = 0
        for layer in layout.layers {
            guard let planLayer = plan.layers.first(where: { $0.layerIndex == layer.layer }) else {
                throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed layer")
            }
            guard layer.experts.count == planLayer.expertsPerLayer,
                  layout.expertStride == planLayer.expertStride else {
                throw RepackError.configurationInvalid(detail:
                    "layout.json validation failed: plan mismatch in layer \(layer.layer)")
            }
            validatedLogicalExperts += layer.experts.count
        }
        guard layout.layers.count == plan.layers.count else {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: layer count mismatch")
        }
        audit?.packedExpertLayoutAuditLogicalIDCount = validatedLogicalExperts
        audit?.packedExpertLayoutOffsetValidationPassed = true
    }
}
