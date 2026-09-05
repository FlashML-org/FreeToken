# White paper release payload

Stage the following files for the white-paper release. The tag must include
the code revision and these files together, but it must not include the local
`tmp/` review images or `.reference-llama-cpp/` reference checkout.

## White paper and metadata

- `paper-draft/amd_strix_halo_freetoken_port_draft.md`
- `paper-draft/paper.tex`
- `paper-draft/references.bib`
- `paper-draft/README.md`
- `paper-draft/RELEASE_NOTES_v0.1.0-rc1.md`
- `paper-draft/PUBLICATION_CHECKLIST.md`
- `paper-draft/RELEASE_PAYLOAD.md`
- `output/pdf/freetoken-amd-strix-halo-white-paper-v0.1.0-rc1.pdf`
- `CITATION.cff`
- `.zenodo.json`

## Public reproduction material

- `docs/amd-rocm-gfx1151.md`
- `docs/reproducibility.md`
- `scripts/build_paper_pdf.py`
- `scripts/reproduce/collect_host_manifest.sh`
- `benchmarks/reproduce/run_local_api_benchmark.py`
- `tests/reproduce/test_collect_host_manifest.py`
- `tests/reproduce/test_run_local_api_benchmark.py`

## Required exclusions

- Model weights and model directories
- Raw service logs and environment dumps
- LAN addresses, hostnames, serial numbers, credentials, and private paths
- `tmp/` renderer output and `.reference-llama-cpp/`
