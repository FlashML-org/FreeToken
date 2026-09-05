# GMKtec EVO-X2 NVFP4 Marlin parity test

## Purpose

This focused test verifies that the production NVFP4 Marlin-style decode GEMV
produces the same result as FreeToken's original LUT-gather decode kernel on
the HIP runtime. It is a numerical correctness gate for the optimization
path, not an end-to-end throughput claim.

## Command and environment

The test ran on the GMKtec EVO-X2 using the source checkout's existing Python
environment and the native HIP runtime:

```text
python -m pytest tests/moe/test_nvfp4_backends.py -k "marlin" --maxfail=1 -q
```

The selected tests include Marlin output comparison against the baseline
kernel and the cache-stomp sequence that reloads experts after a full-layer
prefill. The test module's CUDA marker is satisfied by the available HIP
device through PyTorch's CUDA-compatible device API.

## Result

```text
sss..s                                                                   [100%]
2 passed, 4 skipped, 6 deselected in 4.30s
```

The two executed tests passed. The four skips are unrelated backend variants
or optional dependencies selected by the module and do not invalidate the
Marlin parity result.

## Interpretation

The production Marlin decode path is numerically equivalent to the retained
baseline within the test's specified tolerance and survives the cache
reload-after-prefill scenario. This qualifies it for further real-service
comparison work. It does not prove that the Marlin path is faster than the
current complete serving configuration, so no promotion or TPS claim follows
from this test alone.
