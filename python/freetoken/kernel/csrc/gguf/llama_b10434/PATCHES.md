# Local b10434 delta

- HIP portability wrappers use `USE_ROCM` and the existing FreeToken GGUF dispatch headers.
- `ggml_cuda_pdl_sync`, `ggml_cuda_pdl_lc`, and launch-attribute helpers are stubbed to no-ops;
  stream-ordered launches make this safe for the caller-owned workspace. No PDL mapping is
  claimed until graph capture/replay evidence exists.
- `small_k` template variant is omitted. RDNA3 dispatch forces `use_small_k=false`.
- No `mul_mat_vec_q_moe` or MMID compaction is ported.
