#pragma once

// int32 entries of the scratch that ggml_cuda_launch_mm_ids_helper needs
int64_t ggml_cuda_mm_ids_helper_scratch_size(int n_experts, int n_tokens, int n_expert_used);

// tile_map (ntiles entries of a flat MoE grid of J-column tiles) may be null for the regular grid
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * scratch,
        int2 * tile_map, int J, int ntiles, int jt_none,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
