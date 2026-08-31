# FreeToken AMD ROCm/HIP Port for Strix Halo v0.1.0-rc1

## Technical white paper release candidate

This release candidate packages the evidence-backed technical white paper,
portable host-manifest collector, local-only benchmark client, citation metadata,
and reproducibility guidance for the FreeToken ROCm/HIP port on AMD Strix Halo.

The package is based on the public `amd-rocm-gfx1151` branch tip
`a937862f171900bd5d1d207c8ff59b40a15ce742`, verified on 30 August 2026.
The release-candidate files are not yet included in that public branch. Do not
cite this release candidate as an immutable publication until the final release
checklist is complete and a tag plus Zenodo version DOI exist.

## Included white-paper claims

- Native ROCm/HIP execution is established on the evaluated AMD Strix Halo
  `gfx1151` system.
- Qwen NVFP4 serving passed the documented deterministic canary with the
  reference router at 27.880 mean client-visible decode tokens/s across three
  warm quality-matched runs.
- The same-file Qwen Q4_K_M control measured 50.63 tokens/s with the native
  HIP router and 50.29 tokens/s with the llama.cpp ROCm 10 control.
- The 0.7 percent Q4 margin is bounded to that stated control and is not a
  general engine ranking.
- A faster NVFP4 Triton router is retained as rejected evidence because it
  changed deterministic model output.

## Publication route

Publish the final package as a GitHub release from an immutable tag, then let
Zenodo archive that release and assign the version DOI. Attach the generated
white-paper PDF to the GitHub release. Use the Zenodo version DOI in the final
paper, release page, and any arXiv technical-report submission.

## Not included

Model weights, private model paths, hostnames, LAN addresses, credentials,
serial numbers, unrelated service logs, and raw environment dumps are excluded
from the public release.

## Correspondence

David Bourdeau: davidbourdeau@gmail.com
