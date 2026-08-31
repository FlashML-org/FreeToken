# Final publication checklist

Use this checklist immediately before publishing the public white paper. A
checked box requires fresh evidence, not an assumption based on this release
candidate.

## Repository and provenance

- [ ] Review all staged changes and confirm no unrelated local work is included.
- [ ] Commit the white-paper package, release notes, citation metadata, and
  public reproduction tools together.
- [ ] Record the resulting full commit SHA in the white paper and release notes.
- [ ] Create and push an immutable tag, for example `amd-strix-halo-white-paper-v0.1.0`.
- [ ] Regenerate the PDF from the tagged source and record its SHA-256.

## Privacy and reproducibility

- [ ] Run the manifest collector against a clean native HIP system.
- [ ] Verify all public artifacts omit credentials, LAN addresses, hostnames,
  serial numbers, private model paths, and unrelated logs.
- [ ] Verify model provenance lists publisher, revision, byte count, SHA-256,
  and license without redistributing model weights.
- [ ] Run the public reproduction tests and retain the output in the release
  preparation record.
- [ ] Recheck every quantitative claim against its cited raw artifact.

## Zenodo and public record

- [ ] Connect GitHub to Zenodo and enable the repository.
- [ ] Confirm `.zenodo.json` title, creator, version, license, keywords, and
  description are correct. Zenodo uses this file in preference to `CITATION.cff`
  when both are present.
- [ ] Create the GitHub release from the immutable tag and attach the PDF.
- [ ] Wait for Zenodo processing, verify the record and version DOI, then check
  the archival status.
- [ ] Replace the release-candidate version and DOI-pending language in the
  paper and `CITATION.cff` with the final tag and version DOI.

## Distribution

- [ ] Publish the Zenodo DOI as the canonical citation route.
- [ ] Publish the technical-report PDF in the GitHub release.
- [ ] Optionally submit the same final PDF to arXiv after verifying the current
  subject-category and endorsement rules.
- [ ] Enable a public repository contact route, preferably Issues for
  reproducible reports and Discussions for general questions.
