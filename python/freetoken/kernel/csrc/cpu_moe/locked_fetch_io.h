// GPU side of the locked-layer fetch (locked_fetch.cpp): one publish kernel in place of the meta copies and the doorbell writes, one gather kernel for the staging rows.
// One source for both runtimes like gpu_sync_io.cu; the spin wait is CUDA only.
#pragma once

#include <cstdint>

namespace freetoken::cpu_moe {

// Writes min(*num_indices, max_rows) and that many src rows into the mapped meta, then done = 0 and ready = task.
// A step with nothing to fetch writes done = 1 and rings nothing, so the wait and the gather pass at once.
void locked_publish(uintptr_t stream, const int64_t* num_indices, const int32_t* src_indices, int max_rows, int64_t* meta_count,
                    int32_t* meta_rows, volatile int64_t* ready, volatile int64_t* done, int64_t task);

// Copies staging row i of every bank into slot evict_slots[i]; bank b reads staging_ptrs[b] and writes dst_ptrs[b], row_bytes[b] bytes a row.
// spin: wait for done inside the kernel (no stream memops); otherwise the caller waits on done before the launch.
void locked_gather(uintptr_t stream, const int64_t* num_indices, int max_rows, const int32_t* evict_slots, const int64_t* dst_ptrs,
                   const int64_t* staging_ptrs, const int64_t* row_bytes, int num_banks, int blocks_per_bank, const volatile int64_t* done,
                   bool spin);

}  // namespace freetoken::cpu_moe
