assert __name__ == "__main__"


def generate_clangd():
    import os
    import subprocess

    from freetoken.kernel.utils import DEFAULT_INCLUDE
    from freetoken.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE

    # TODO(ROCm): hiprtc JIT cache should be separate from nvcc JIT cache to avoid stale binaries.
    try:
        status = subprocess.run(
            args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            check=True,
        )
        compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
        major, minor = compute_cap.split(".")
        arch_flags = ["-xcuda", f"--cuda-gpu-arch=sm_{major}{minor}"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # TODO(ROCm): parse gfx target from rocm-smi; default to gfx1100 for now.
        arch_flags = ["-xhip", "--offload-arch=gfx1100"]
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
