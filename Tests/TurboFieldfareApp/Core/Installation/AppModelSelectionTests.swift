import Foundation
import Testing
import TurboFieldfare

@testable import TurboFieldfareAppCore

/// The model-selection path had no coverage at all: nothing referenced
/// `TURBO_FIELDFARE_MODEL`, `.selected`, or `installDirectoryName`. That is the
/// same shape as the `layout.json` cap bug -- Qwen support added on one path,
/// missed on another, invisible until someone ran it against a real pack.
@Suite struct AppModelSelectionTests {

    @Test("Each family maps to its own descriptor, directory, and repo")
    func descriptorsAreDistinctPerFamily() {
        let gemma = AppModelInstallDescriptor.descriptor(for: .gemma4)
        let qwen = AppModelInstallDescriptor.descriptor(for: .qwen36)

        #expect(gemma == .default)
        #expect(qwen == .qwen36)
        #expect(gemma?.installDirectoryName == "gemma4.gturbo")
        #expect(qwen?.installDirectoryName == "qwen36.gturbo")
        // Distinct repos: a wrong mapping here downloads the other model.
        #expect(gemma?.repoID != qwen?.repoID)
    }

    /// `family` is the inverse of `descriptor(for:)`; if they disagree the
    /// picker shows one model selected while another is downloaded.
    @Test("descriptor(for:) and .family round-trip for every family")
    func familyRoundTrips() {
        for family in AppModelInstallDescriptor.selectableFamilies {
            let descriptor = AppModelInstallDescriptor.descriptor(for: family)
            #expect(descriptor?.family == family,
                    "descriptor(for: \(family.rawValue)) does not report that family back")
        }
    }

    @Test("Every selectable family has a descriptor to install")
    func everySelectableFamilyIsInstallable() {
        for family in AppModelInstallDescriptor.selectableFamilies {
            #expect(AppModelInstallDescriptor.descriptor(for: family) != nil,
                    "\(family.rawValue) is offered in the picker with no descriptor")
        }
    }

    /// Only Gemma ships an image tower. Pairing Qwen with Gemma's companion
    /// would put one family's vision pack on another family's text model.
    @Test("Only Gemma offers a vision companion")
    func visionCompanionIsGemmaOnly() {
        #expect(AppModelInstallDescriptor.visionCompanion(for: .gemma4) != nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: .qwen36) == nil)
    }

    @Test("The install location follows the selected descriptor")
    func locationFollowsDescriptor() {
        let gemma = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { $0 == "/repo/Package.swift" || $0 == "/repo/Sources/TurboFieldfareApp/Mac" },
            installDirectoryName: AppModelInstallDescriptor.default.installDirectoryName)
        let qwen = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo", isDirectory: true),
            applicationSupportURL: URL(fileURLWithPath: "/support", isDirectory: true),
            fileExists: { $0 == "/repo/Package.swift" || $0 == "/repo/Sources/TurboFieldfareApp/Mac" },
            installDirectoryName: AppModelInstallDescriptor.qwen36.installDirectoryName)

        #expect(gemma.path == "/repo/scratch/gemma4.gturbo")
        #expect(qwen.path == "/repo/scratch/qwen36.gturbo")
        #expect(gemma.path != qwen.path,
                "both families would install over each other")
    }

    @MainActor
    @Test("Selecting a family moves the model directory and the descriptor")
    func selectingAFamilySwitchesBoth() throws {
        let model = AppModel(
            modelDirectory: AppModelLocation.defaultURL(
                descriptor: AppModelInstallDescriptor.default),
            installer: RepackModelInstallerClient(descriptor: .default))
        #expect(model.selectedModelFamily == .gemma4)

        model.selectModelFamily(.qwen36)

        #expect(model.selectedModelFamily == .qwen36)
        #expect(model.installDescriptor == .qwen36)
        #expect(model.modelPathText.hasSuffix("qwen36.gturbo"),
                "the model directory did not follow the selection: \(model.modelPathText)")
    }

    @MainActor
    @Test("Selecting the family already chosen is a no-op")
    func reselectingIsANoOp() throws {
        let model = AppModel(
            modelDirectory: AppModelLocation.defaultURL(
                descriptor: AppModelInstallDescriptor.default),
            installer: RepackModelInstallerClient(descriptor: .default))
        let before = model.modelPathText

        model.selectModelFamily(.gemma4)

        #expect(model.selectedModelFamily == .gemma4)
        #expect(model.modelPathText == before)
    }

    /// Switching mid-transfer would strand a partial download owned by a
    /// descriptor that is no longer selected.
    @MainActor
    @Test("Switching is refused while a transfer is in flight")
    func switchingIsRefusedDuringInstall() throws {
        let model = AppModel(
            modelDirectory: AppModelLocation.defaultURL(
                descriptor: AppModelInstallDescriptor.default),
            installer: RepackModelInstallerClient(descriptor: .default))
        #expect(model.canSelectModelFamily)

        model.installState = .checking

        #expect(!model.canSelectModelFamily)
        model.selectModelFamily(.qwen36)
        #expect(model.selectedModelFamily == .gemma4,
                "a switch landed while a download was in flight")
    }
}
