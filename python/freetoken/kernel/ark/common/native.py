"""Source-only HIP build, keyed by source, compiler, Torch and architecture."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]
_LOADED = False
_LOCK = threading.Lock()


def source_digest(native_only=False):
    suffixes = (".hip", ".h", ".cpp") if native_only else (".hip", ".h", ".cpp", ".py")
    paths = sorted(p for p in ROOT.rglob("*") if p.suffix in suffixes and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _compiler():
    from torch.utils.cpp_extension import ROCM_HOME

    if os.environ.get("HIPCC"):
        return os.environ["HIPCC"]
    for root in (ROCM_HOME, os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if root and (Path(root) / "bin/hipcc").is_file():
            return str(Path(root) / "bin/hipcc")
    return shutil.which("hipcc") or "/opt/rocm/bin/hipcc"


def cache_dir():
    import torch

    compiler = _compiler()
    version = subprocess.check_output([compiler, "--version"])
    recipe = Path(__file__).read_bytes()
    key = hashlib.sha256(
        source_digest(native_only=True).encode() + recipe + str(torch.__version__).encode() + str(torch.version.hip).encode() + version
    ).hexdigest()[:20]
    root = Path(os.environ.get("FREETOKEN_ARK_CACHE_DIR", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "freetoken/ark"))
    return root / key


def library_path():
    return cache_dir() / "libfreetoken_ark.so"


def load():
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        import torch

        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("ARK requires ROCm and a gfx1201 GPU")
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1201"):
            raise RuntimeError("ARK kernels are compiled for gfx1201 only")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call freetoken.kernel.ark.initialize() before HIP Graph capture")
        build = cache_dir()
        build.mkdir(parents=True, exist_ok=True)
        with (build / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lib = build / "libfreetoken_ark.so"
            if not lib.exists():
                compiler = _compiler()
                sources = sorted(ROOT.glob("*/*.hip"))
                flags = ["-O3", "-std=c++17", "--offload-arch=gfx1201", "-fPIC", "-I" + str(ROOT / "common/include")]
                objects = []
                for source in sources:
                    obj = build / (source.parent.name + "_" + source.stem + ".o")
                    command = [compiler, *flags, "-c", str(source), "-o", str(obj)]
                    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    (build / (obj.stem + ".log")).write_text(proc.stdout)
                    if proc.returncode:
                        raise RuntimeError(f"ARK HIP build failed: {command}\n{proc.stdout}")
                    objects.append(obj)
                subprocess.run([compiler, "-shared", *map(str, objects), "-o", str(lib)], check=True)
                (build / "manifest.json").write_text(
                    json.dumps(
                        {
                            "source_sha256": source_digest(),
                            "torch": str(torch.__version__),
                            "hip": str(torch.version.hip),
                            "compiler": subprocess.check_output([compiler, "--version"], text=True),
                            "flags": flags,
                            "library_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
                        },
                        indent=2,
                    )
                )
            # hipify may generate a sibling source; keep installed packages read-only.
            binding = build / "bindings.cpp"
            shutil.copyfile(ROOT / "common/bindings.cpp", binding)
            from torch.utils.cpp_extension import load as cpp_load

            cpp_load(
                name="freetoken_ark_ops_" + build.name,
                sources=[str(binding)],
                extra_cflags=["-O3"],
                extra_ldflags=[f"-L{build}", "-lfreetoken_ark", f"-Wl,-rpath,{build}"],
                build_directory=str(build),
                with_cuda=True,
                is_python_module=False,
                verbose=os.environ.get("FREETOKEN_ARK_BUILD_VERBOSE") == "1",
            )
        _LOADED = True


if __name__ == "__main__":
    load()
    print(library_path())
