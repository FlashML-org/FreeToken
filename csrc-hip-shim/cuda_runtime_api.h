// HIP compatibility shim for FreeToken's host-side extensions.
// Maps every CUDA runtime API symbol used by kernel/csrc onto its HIP
// equivalent so the same sources build against ROCm without edits.
#pragma once
#include <hip/hip_runtime_api.h>

#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaGetErrorString hipGetErrorString
#define cudaStream_t hipStream_t

#define cudaMallocHost hipMallocHost
#define cudaHostAlloc hipHostAlloc
#define cudaFreeHost hipFreeHost
#define cudaHostRegister hipHostRegister
#define cudaHostGetDevicePointer hipHostGetDevicePointer

#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped

#define cudaGetDevice hipGetDevice
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaLaunchHostFunc hipLaunchHostFunc

#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem

#define CUDART_CB
