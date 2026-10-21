// GPU side of the locked-layer fetch (locked_fetch.cpp): one publish kernel in place of the meta copies and the doorbell writes.
// One source for both runtimes like gpu_sync_io.cu.
#pragma once

#include <cstdint>

namespace freetoken::cpu_moe {

// Writes min(*num_indices, max_rows) and that many src rows into the mapped meta, then done = 0 and ready = task.
// A step with nothing to fetch writes done = 1 and rings nothing, so the wait and the gather pass at once.
void locked_publish(uintptr_t stream, const int64_t* num_indices, const int32_t* src_indices, int max_rows, int64_t* meta_count,
                    int32_t* meta_rows, volatile int64_t* ready, volatile int64_t* done, int64_t task);

}  // namespace freetoken::cpu_moe
