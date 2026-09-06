from freetoken.models.gguf import reader


def test_page_releaser_trims_only_after_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(reader, "release_mapped_pages", calls.append)

    releaser = reader.PageReleaser("model.gguf", budget=10)
    releaser.note(4)
    releaser.note(5)
    assert calls == []

    releaser.note(1)
    assert calls == ["model.gguf"]


def test_page_releaser_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(reader, "release_mapped_pages", calls.append)

    releaser = reader.PageReleaser("model.gguf", budget=0)
    releaser.note(1 << 40)
    assert calls == []
