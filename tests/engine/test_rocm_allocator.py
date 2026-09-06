def test_windows_rocm_leaves_expandable_segments_off(monkeypatch):
    import freetoken.engine.engine as engine

    called = []
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(engine.sys, "platform", "win32")
    monkeypatch.setattr(engine.torch.version, "hip", "test-rocm")
    monkeypatch.setattr(
        engine.torch.cuda.memory,
        "_set_allocator_settings",
        lambda value: called.append(value),
    )

    engine._ensure_expandable_segments()

    assert called == []


def test_explicit_allocator_config_wins_on_windows_rocm(monkeypatch):
    import freetoken.engine.engine as engine

    called = []
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(engine.sys, "platform", "win32")
    monkeypatch.setattr(engine.torch.version, "hip", "test-rocm")
    monkeypatch.setattr(
        engine.torch.cuda.memory,
        "_set_allocator_settings",
        lambda value: called.append(value),
    )

    engine._ensure_expandable_segments()

    assert called == []
