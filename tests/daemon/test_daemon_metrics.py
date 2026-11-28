"""The engine's VRAM footprint off NVIDIA. Every probe in ``freetoken.daemon.metrics`` is best-effort
and answers 0 when it cannot read the GPU, so a break in the DRM fdinfo path raises nothing: an AMD
engine just shows as using no VRAM."""

from __future__ import annotations

from freetoken.daemon import metrics

DGPU = "0000:03:00.0"
IGPU = "0000:0e:00.0"


def _fdinfo(pdev: str, client: int, vram: str, key: str = "drm-resident-vram") -> str:
    return (
        "pos:\t0\nflags:\t02100002\nmnt_id:\t27\nino:\t1146\n"
        f"drm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-pdev:\t{pdev}\n"
        f"drm-total-vram:\t{vram}\n{key}:\t{vram}\ndrm-memory-gtt: \t8260 KiB\n"
    )


def _fake_proc(tmp_path, monkeypatch, procs: dict[int, dict[int, str]]) -> None:
    for pid, fds in procs.items():
        fdinfo = tmp_path / str(pid) / "fdinfo"
        fdinfo.mkdir(parents=True)
        for fd, text in fds.items():
            (fdinfo / str(fd)).write_text(text)
    monkeypatch.setattr(metrics, "_PROC", str(tmp_path))
    monkeypatch.setattr(metrics, "_nvml_process_vram", lambda: None)
    monkeypatch.setattr(metrics, "_smi_process_vram", lambda: {})


def test_vram_falls_back_to_drm_fdinfo(tmp_path, monkeypatch):
    plain_file = "pos:\t0\nflags:\t0100002\nmnt_id:\t30\nino:\t77\n"
    worker = {
        9: _fdinfo(DGPU, 313, "15034484 KiB"),
        10: _fdinfo(IGPU, 314, "280 KiB"),
        11: _fdinfo(DGPU, 313, "15034484 KiB"),  # dup of fd 9
        12: plain_file,
    }
    _fake_proc(
        tmp_path,
        monkeypatch,
        {
            100: {0: plain_file},
            101: worker,
            102: {9: worker[9]},  # forked child, same client as its parent
            555: {9: _fdinfo(DGPU, 400, "1024 KiB")},  # another program on the same card
        },
    )
    # pid 103 has exited between the tree walk and the read
    assert metrics.vram_bytes_for_pids([100, 101, 102, 103]) == (15034484 + 280) * 1024


def test_drm_fdinfo_reads_the_older_key_and_other_units(tmp_path, monkeypatch):
    _fake_proc(
        tmp_path,
        monkeypatch,
        {
            1: {3: _fdinfo(DGPU, 1, "2048 KiB", key="drm-memory-vram")},
            2: {3: _fdinfo(DGPU, 2, "3 MiB")},
            3: {3: _fdinfo(DGPU, 3, "512")},
        },
    )
    assert metrics.vram_bytes_for_pids([1, 2, 3]) == 2048 * 1024 + 3 * (1 << 20) + 512
