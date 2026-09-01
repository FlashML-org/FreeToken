"""CPU-only AMD/HIP detection. No GPU is required; sysfs/HIP are mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.runtime import gpu


@pytest.fixture(autouse=True)
def _clear_vendor_cache():
    gpu.vendor.cache_clear()
    yield
    gpu.vendor.cache_clear()


def test_igpu_hides_granite_ridge_not_strix_halo():
    assert gpu.is_igpu(name="AMD Radeon Graphics", arch="gfx1036", total_bytes=512 << 20)
    assert gpu.is_igpu(name="AMD Radeon Graphics (Granite Ridge)", arch=None)
    assert not gpu.is_igpu(
        name="AMD Radeon AI PRO R9700", arch="gfx1201", total_bytes=32 << 30
    )
    assert not gpu.is_igpu(name="AMD Radeon 8060S", arch="gfx1151", total_bytes=128 << 30)


def test_arch_from_name_r9700():
    assert gpu._arch_from_name("AMD Radeon AI PRO R9700") == "gfx1201"
    assert gpu._arch_from_name("Radeon RX 9070 XT") == "gfx1201"
    assert gpu._arch_from_name("Radeon RX 9060 XT") == "gfx1200"


def test_vendor_forced(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    assert gpu.vendor() is gpu.Vendor.AMD
    assert gpu.is_hip()
    assert not gpu.is_cuda()


def test_vendor_cuda_wheel_plus_igpu_sysfs_is_nvidia(monkeypatch, tmp_path):
    """9950X Granite Ridge (0x1002 iGPU) + CUDA wheel is a normal NVIDIA box."""
    drm = tmp_path / "card0" / "device"
    drm.mkdir(parents=True)
    (drm / "vendor").write_text("0x1002\n")
    (drm / "boot_vga").write_text("1\n")
    (drm / "mem_info_vram_total").write_text(str(512 << 20) + "\n")
    monkeypatch.delenv("FREETOKEN_GPU_VENDOR", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("FREETOKEN_INCLUDE_IGPU", raising=False)
    monkeypatch.setattr(gpu, "_hip_library_name", lambda: None)
    monkeypatch.setattr(gpu, "_torch", lambda: SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0")))
    monkeypatch.setattr(gpu, "hip_enumerate_devices", lambda: [])
    orig_sysfs = gpu._sysfs_amd_devices
    monkeypatch.setattr(gpu, "_sysfs_amd_devices", lambda drm_root="/sys/class/drm": orig_sysfs(str(tmp_path)))
    gpu.vendor.cache_clear()
    assert gpu.vendor() is gpu.Vendor.NVIDIA
    assert gpu.require_gpu() is gpu.Vendor.NVIDIA
    # CUDA_VISIBLE_DEVICES=0 is an NVIDIA ordinal, not drm card0 (the iGPU).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    gpu.vendor.cache_clear()
    assert gpu.vendor() is gpu.Vendor.NVIDIA


def test_vendor_cuda_wheel_on_discrete_amd_is_cuda_on_amd(monkeypatch):
    monkeypatch.delenv("FREETOKEN_GPU_VENDOR", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "_hip_library_name", lambda: None)
    monkeypatch.setattr(gpu, "_torch", lambda: SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0")))
    monkeypatch.setattr(
        gpu,
        "hip_enumerate_devices",
        lambda: [
            {
                "index": 0,
                "name": "AMD Radeon AI PRO R9700",
                "arch": "gfx1201",
                "total_bytes": 32 << 30,
                "hidden_igpu": False,
            }
        ],
    )
    gpu.vendor.cache_clear()
    assert gpu.vendor() is gpu.Vendor.CUDA_ON_AMD
    with pytest.raises(RuntimeError, match="ZLUDA"):
        gpu.require_gpu()


def test_vendor_cuda_wheel_hip_visible_discrete_amd(monkeypatch):
    monkeypatch.delenv("FREETOKEN_GPU_VENDOR", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "_hip_library_name", lambda: None)
    monkeypatch.setattr(gpu, "_torch", lambda: SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0")))
    monkeypatch.setattr(
        gpu,
        "hip_enumerate_devices",
        lambda: [
            {"index": 0, "name": "AMD Radeon Graphics", "arch": "gfx1036", "hidden_igpu": True},
            {"index": 1, "name": "AMD Radeon AI PRO R9700", "arch": "gfx1201", "hidden_igpu": False},
        ],
    )
    gpu.vendor.cache_clear()
    assert gpu.vendor() is gpu.Vendor.CUDA_ON_AMD


def test_sysfs_igpu_not_discrete(tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "boot_vga").write_text("1\n")
    (card / "mem_info_vram_total").write_text(str(512 << 20) + "\n")
    devs = gpu._sysfs_amd_devices(str(tmp_path))
    assert len(devs) == 1
    assert devs[0]["hidden_igpu"] is True


def test_require_gpu_none(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "none")
    gpu.vendor.cache_clear()
    with pytest.raises(RuntimeError, match="No usable GPU"):
        gpu.require_gpu()


def test_nvidia_only_error_mentions_triton():
    err = gpu.nvidia_only_error("flashinfer")
    assert "NVIDIA-only" in str(err)
    assert "triton" in str(err).lower()
    assert "ZLUDA" in str(err)


def test_apply_amd_runtime_env_sets_arch(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    monkeypatch.delenv("TVM_FFI_ROCM_ARCH_LIST", raising=False)
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("FREETOKEN_GFX_ARCH", raising=False)
    monkeypatch.setattr(gpu, "hip_enumerate_devices", lambda: [])
    gpu.vendor.cache_clear()
    written = gpu.apply_amd_runtime_env()
    assert written["TVM_FFI_ROCM_ARCH_LIST"] == "gfx1201"
    assert written["TRITON_OVERRIDE_ARCH"] == "gfx1201"


def test_apply_hides_igpu_from_visible_devices(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("FREETOKEN_INCLUDE_IGPU", raising=False)
    monkeypatch.setattr(
        gpu,
        "hip_enumerate_devices",
        lambda: [
            {"index": 0, "name": "AMD Radeon Graphics", "arch": "gfx1036", "hidden_igpu": True},
            {"index": 1, "name": "R9700", "arch": "gfx1201", "hidden_igpu": False},
            {"index": 2, "name": "R9700", "arch": "gfx1201", "hidden_igpu": False},
        ],
    )
    gpu.vendor.cache_clear()
    written = gpu.apply_amd_runtime_env()
    assert written["HIP_VISIBLE_DEVICES"] == "1,2"
    assert written["CUDA_VISIBLE_DEVICES"] == "1,2"
    ranks = gpu.tp_rank_devices(
        2,
        [
            {"index": 0, "name": "AMD Radeon Graphics", "arch": "gfx1036", "hidden_igpu": True},
            {"index": 1, "name": "R9700", "arch": "gfx1201", "hidden_igpu": False},
            {"index": 2, "name": "R9700", "arch": "gfx1201", "hidden_igpu": False},
        ],
    )
    assert [d["index"] for d in ranks] == [1, 2]


def test_graph_replay_unsafe_on_gfx1201(monkeypatch):
    monkeypatch.delenv("FREETOKEN_HIP_GRAPH_REPLAY", raising=False)
    assert gpu.hip_graph_replay_safe("gfx1201") is False
    assert gpu.hip_graph_replay_safe("gfx1200") is True
    monkeypatch.setenv("FREETOKEN_HIP_GRAPH_REPLAY", "1")
    assert gpu.hip_graph_replay_safe("gfx1201") is True


def test_hip_triton_alias_maps_dsv4_mla_bsa():
    from freetoken.attention import AttnType, hip_triton_alias

    assert hip_triton_alias(frozenset({AttnType.DSV4})) == "dsv4_sparse"
    assert hip_triton_alias(frozenset({AttnType.MLA})) == "dsa"
    assert hip_triton_alias(frozenset({AttnType.BSA})) == "m3_sparse"
    assert hip_triton_alias(frozenset({AttnType.FULL})) == "triton"


def test_describe_lists_zluda_unsupported(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "none")
    gpu.vendor.cache_clear()
    text = gpu.describe()
    assert "zluda: unsupported" in text
    assert "vendor: none" in text
