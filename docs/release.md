# Releasing

Each release carries a CycloneDX SBOM (Software Bill of Materials): a machine-readable list of every
package pinned in `uv.lock`, with versions and hashes. Users scan it for known vulnerabilities
(`grype sbom:sbom-<version>.cyclonedx.json`) or audit what a release ships, without installing it.

The repository enforces **immutable releases**: once a release is published, its assets and tag are
locked, and a deleted release's tag name can never be reused. The SBOM is therefore attached while the
release is still a draft. `.github/workflows/release.yml` does that when a version tag is pushed.

## Steps

1. Bump `version` in `pyproject.toml` and the `dna-factory` entry in `uv.lock`, commit
   (`chore: bump version to X.Y.Z`), and push `master`.
2. Tag and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. The **Release SBOM** workflow checks that the tag matches `pyproject.toml`, generates
   `sbom-X.Y.Z.cyclonedx.json` with Syft, and creates a draft release with auto-generated notes and
   the SBOM plus its `.sha256` attached.
4. Edit the draft's notes, check the assets, and publish:

   ```bash
   gh release edit vX.Y.Z --notes-file notes.md --draft=false --latest
   ```

Do not publish before the workflow finishes: a release published without the SBOM cannot get it
afterwards. If the workflow failed, fix it and rerun it on the draft from the Actions tab
(**Run workflow** with the tag) before publishing.
