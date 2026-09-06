# AMD ROCm target matrix

FreeToken uses exact `gcnArchName` target selection. It never substitutes a nearby
gfx target when requested target is absent from matrix.

| target | family | wave | native GGUF types | evidence |
| --- | --- | ---: | --- | --- |
| gfx1100 | RDNA3 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | direct native tests; Qwen3.5 CPU/GPU one-token serving smoke |
| gfx1101 | RDNA3 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1102 | RDNA3 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1103 | RDNA3 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1150 | RDNA3.5 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1151 | RDNA3.5 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1200 | RDNA4 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |
| gfx1201 | RDNA4 | 32 | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 | compile-only |

`compile-only` means source/toolchain selection is present. It does not mean a
served model completed successfully. Promote one cell only after fresh build,
eager/graph result, finite logits, non-empty exact completion, and native/reference
comparison are recorded for that physical target.

ROCm 7.2.1 currently fails closed for CPU/Hybrid HIP graph memop capture in this
checkout. Eager execution remains available; graph replay requires a HIP runtime
with tested signal-memory graph nodes.
