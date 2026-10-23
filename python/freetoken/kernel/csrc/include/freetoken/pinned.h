#pragma once

#include <freetoken/hip_compat.h>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>

namespace host {

// Pinned host memory is GPU-dereferenceable at its host VA only where UVA identity
// holds (Linux). On Windows/WDDM, cudaHostRegister'd memory maps to a different device
// address, so host-resident tensors are translated where their pointer enters kernel params.
// Cached once per process: FreeToken pins one CUDA device per process (set at engine launch).
inline bool host_ptr_identity() {
    static const bool identity = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;  // fail closed: translate (and surface errors), don't assume identity
        }
        int uva = 0, reg = 0;
        cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
        cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
        return uva == 1 && reg == 1;
    }();
    return identity;
}

inline void* device_alias(void* ptr, DLDevice dev) {
    if (dev.device_type == kDLCUDA || dev.device_type == kDLROCM || host_ptr_identity()) {
        return ptr;
    }
    void* mapped = nullptr;
    const auto err = cudaHostGetDevicePointer(&mapped, ptr, 0);
    RuntimeCheck(err == cudaSuccess,
        "host tensor must be pinned+mapped (cudaHostGetDevicePointer: ",
        cudaGetErrorString(err), ")");
    return mapped;
}

}  // namespace host
