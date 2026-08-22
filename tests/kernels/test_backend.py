def test_native_wheels_are_disabled_on_turing(monkeypatch):
    import torch
    from freetoken.kernel import backend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    backend.is_flashinfer_installed.cache_clear()
    backend.is_sgl_kernel_installed.cache_clear()
    backend.is_triton_kernels_installed.cache_clear()

    try:
        assert not backend.is_flashinfer_installed()
        assert not backend.is_sgl_kernel_installed()
        assert not backend.is_triton_kernels_installed()
    finally:
        backend.is_flashinfer_installed.cache_clear()
        backend.is_sgl_kernel_installed.cache_clear()
        backend.is_triton_kernels_installed.cache_clear()
