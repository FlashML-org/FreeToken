# Vendored Triton GGUF attribution

FreeToken includes 52 adapted files from
[vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
under the Apache License, Version 2.0. The upstream license is reproduced
without alteration in [LICENSE](LICENSE). FreeToken's root license is retained.

## Source and authorship

The verified upstream content baseline is
`a494069684fcbfb8c23aff8266d9bceea7569b57` (2026-06-11). GEMM was introduced by Isotr0py in
[PR #13](https://github.com/vllm-project/vllm-gguf-plugin/pull/13), merged as
`92a01df7f93724ad2c2ba28144a9742527295a1d`; dequantize was introduced
by Isotr0py in [PR #24](https://github.com/vllm-project/vllm-gguf-plugin/pull/24).
This credits the publicly documented contributions without claiming exclusive
authorship or assigning copyright ownership.

Maxritz adapted these sources for FreeToken in
[donor commit 45675e4](https://github.com/Maxritz/FreeToken-ROCm/commit/45675e47348a167a9b36ca224ad310c1ee1e34b4).
All 52 public donor files match the local donor Git blobs byte for byte.
The local production import is `f3cec5ab9a87ddfab9d2961287e1952fbebd4664`.
Existing commit authorship is retained.

The donor's actual upstream checkout revision is unknown. The baseline above
is reconstructed from verified public content, not a claim about that checkout.
The same 52 upstream blobs and LICENSE also occur at
`fb973ad784f38b98b054e136bec3414b7cd8494d`, the checked snapshot preceding the donor commit.
[PROVENANCE.json](PROVENANCE.json) records this uncertainty explicitly.

## Scope and modifications

- `vllm_gguf_plugin/triton/dequantize/` maps to
  `python/freetoken/kernel/triton/gguf_dequant/` (25 files).
- `vllm_gguf_plugin/triton/gemm/` maps to
  `python/freetoken/kernel/triton/gguf_gemm/` (27 files).

In the donor, 23 files were copied exactly; 22 dequantize files had relative
GEMM imports relocated and one empty EOF line added; three dequantize package
initializers had their `__all__` declarations removed; and four otherwise
empty GEMM initializers lost the upstream SPDX line. The production import
removed those 22 empty EOF lines. This qualification restores the four SPDX
lines and adds license/attribution/modification comments to the modified files.

Iskandar added row-aware decode tiling to `gguf_gemm/utils.py` and
`gguf_gemm/standard_quant/q4_0.py` on 2026-09-05 in
`845e541d70040e610dde7231c34e0e89358c64b8`. This qualification makes no executable Python changes.

The upstream `dequantize/__init__.py` is not among the 52 vendored files.
Fused-MoE and native C/CUDA sources in the plugin are outside this import.

## License and NOTICE decision

Both complete checked upstream archives contain a root Apache-2.0 LICENSE and
no NOTICE or nested license file. There is no upstream NOTICE from these
snapshots to reproduce. This attribution document records source credit and
modifications; it is not an upstream NOTICE and does not change license terms.
The four recovered SPDX lines are the only individual license headers present
in the 52 corresponding upstream files.

`gguf_gemm/iq_quant/iq_tables.py` is an exact upstream copy. It obtains
table values at runtime from the separately installed `gguf.quants`
dependency. This import does not vendor the dependency's implementation or
the plugin's native GGML headers, and does not relicense those dependencies.

## Hashes and packaging

The manifest distinguishes upstream, donor, production-import,
production-baseline and post-annotation source hashes. Source SHA-256 values
use Git blob bytes (UTF-8 with LF line endings); CRLF checkout bytes must be
converted to LF before comparison. Remote LICENSE SHA-256 uses its exact bytes.
The manifest covers these 52 files, not a repository-wide license audit.

The project's `license-files` list includes this document, LICENSE and
PROVENANCE.json so that source distributions and wheels retain the provenance
materials. Packaging configuration was checked statically; no distribution
build or runtime/GPU validation is implied.
