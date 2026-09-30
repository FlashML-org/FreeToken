# AMD ROCm 10 installation and qualification

FreeToken's AMD path is being qualified on the ROCm 10 stack. The supported
campaign environment uses ROCm `10.0.0` and a matching ROCm-enabled PyTorch
distribution; this is separate from the CUDA `accel` extra and its NVIDIA
packages.

## Runtime policy

- Use ROCm 10 only. Do not install ROCm 7.x packages, wheels, SDKs, or indexes.
- On a qualified target, `/opt/rocm` selects `/opt/rocm-10.0`, the
  version file reports `10.0.0`, and AMD ROCm packages are release `10.0.0-4`.
- The validated Python environment uses PyTorch `2.13.0+rocm10.0.0`. Its HIP
  component version string is `7.15.26333`; treat this as component metadata,
  not evidence of a separately installed ROCm 7.x distribution.
- Keep the system ROCm installation, FreeToken source, HIP Python environment,
  test evidence, and model files separate. Do not replace the system runtime
  or reuse a CUDA-enabled environment during candidate qualification.

AMD's [official ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html)
documents supported system installation methods. Qualification targets with
ROCm 10 already installed should use that installation and
must not rerun a system installer unless a separately authorized remediation
requires it.

## FreeToken source build

Provision the matching ROCm 10 PyTorch/HIP/Triton environment before building
FreeToken. Build the exact candidate from its isolated checkout with package
resolution disabled for that step; this prevents PEP 517 isolation or pip from
replacing the pre-provisioned ROCm 10 ABI. The repository's `rocm` extra is
intentionally empty and does not install or choose an AMD toolchain version.

The candidate build is not device qualification. Record the exact source SHA,
system ROCm version and package release, PyTorch and HIP versions, `gfx` target,
native build/import result, and the full focused-test accounting. A run with
GPU visibility hidden only covers host-side behavior; HIP kernels, inference,
and throughput require separate device-visible acceptance tests on each target.

```bash
# Install the editable FreeToken source without replacing the pre-provisioned ROCm 10 packages.
python -m pip install --no-deps --no-build-isolation -e .
```

See [ROCm 10 runtime and qualification notes](amd-rocm-gfx1151.md) for the
campaign's tested environment layout and evidence boundaries. The current
architecture targets are `gfx1151` and `gfx1150`; evidence
for one target must not be generalized to another.
