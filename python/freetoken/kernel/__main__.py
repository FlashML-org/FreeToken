assert __name__ == "__main__"


def generate_clangd():
    import os
    import subprocess

    from freetoken.kernel.utils import DEFAULT_INCLUDE
    from freetoken.utils import get_rocm_gfx_arch, init_logger, is_rocm
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE

    # TODO(ROCm): hiprtc JIT cache should be separate from nvcc JIT cache to avoid stale binaries.
    if is_rocm():
        arch_flags = ["-xhip", f"--offload-arch={get_rocm_gfx_arch() or 'gfx1201'}"]
    else:
        try:
            status = subprocess.run(
                args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                check=True,
            )
            compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
            major, minor = compute_cap.split(".")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            import torch

            major, minor = torch.cuda.get_device_capability()
        arch_flags = ["-xcuda", f"--cuda-gpu-arch=sm_{major}{minor}"]
    compile_flags = ",\n    ".join(
        arch_flags + ["-std=c++20", "-Wall", "-Wextra"]
        + [f"-isystem{path}" for path in include_paths]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()
