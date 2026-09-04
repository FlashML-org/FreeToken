"""mRoPE for Qwen3.8-Flash-Next image prompts.

Ports two pieces of HF ``modeling_qwen4_exp``: ``get_rope_index`` (the 3-D T/H/W positions of a
prompt and the delta the text after the last image continues from) and
``Qwen4ExpTextRotaryEmbedding.apply_interleaved_mrope`` (the per-token cos|sin row). Text-only
prompts reduce to ``positions`` on all three axes and a zero delta, so the serving path only
builds a table for prefill batches that carry images; decode reads the normal cache at
``position + delta``.

The cos|sin table has the same ``[rows, rotary_dim]`` layout as ``RotaryEmbedding._cos_sin_cache``
(cos on the first half, sin on the second), so the flashinfer/triton rope kernels and the QSA
indexer's ``qsa_index_norm_rope`` take it as the cache with ``positions = arange(rows)``.
"""

from __future__ import annotations

import torch


def rope_index(
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor | None,
    image_token_id: int,
    merge: int,
) -> tuple[torch.Tensor, int]:
    """``([3, L] int64 positions, delta)`` for one prompt (HF ``get_rope_index``, images only).

    Text runs advance all three axes by their length. An image of ``t x h x w`` patches takes
    ``t * (h // merge) * (w // merge)`` placeholder tokens at T/H/W offsets from the current
    position and advances it by ``max(h, w) // merge``. ``delta`` is what decode adds to a
    token index to get its rope position (``max_pos + 1 - L``, <= 0).
    """
    ids = input_ids.tolist()
    length = len(ids)
    out = torch.empty(3, length, dtype=torch.int64)
    cur = i = img = 0
    while i < length:
        if ids[i] == image_token_id:
            t, h, w = (int(v) for v in grid_thw[img])
            img += 1
            hh, ww = h // merge, w // merge
            n = t * hh * ww
            assert ids[i : i + n] == [image_token_id] * n, (
                "image placeholder run is too short"
            )
            tt, hp, wp = torch.meshgrid(
                torch.arange(t), torch.arange(hh), torch.arange(ww), indexing="ij"
            )
            out[0, i : i + n] = tt.reshape(-1) + cur
            out[1, i : i + n] = hp.reshape(-1) + cur
            out[2, i : i + n] = wp.reshape(-1) + cur
            cur += max(hh, ww)
            i += n
            continue
        j = i
        while j < length and ids[j] != image_token_id:
            j += 1
        out[:, i:j] = torch.arange(cur, cur + (j - i))
        cur += j - i
        i = j
    assert grid_thw is None or img == len(grid_thw), "more images than placeholder runs"
    return out, int(out.max()) + 1 - length


def mrope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, section: tuple[int, ...]
) -> torch.Tensor:
    """``[T, 2 * len(inv_freq)]`` fp32 cos|sin rows for 3-D ``positions [3, T]``.

    Interleaved layout (HF ``apply_interleaved_mrope``): frequency ``k`` rotates by the T axis,
    except ``k = 1 + 3m`` (m < section[1]) which use H and ``k = 2 + 3m`` (m < section[2]) W.
    """
    freqs = positions.to(inv_freq.dtype).unsqueeze(-1) * inv_freq  # [3, T, n]
    f = freqs[0].clone()
    for axis in (1, 2):
        idx = slice(axis, section[axis] * 3, 3)
        f[:, idx] = freqs[axis][:, idx]
    return torch.cat((f.cos(), f.sin()), dim=-1)


def mrope_table(
    reqs,
    ratio: int,
    inv_freq: torch.Tensor,
    section: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rope rows for a prefill batch that carries image tokens: ``(rope_positions [T] int32 = each
    token's row, table [rows, rotary_dim] fp32 cos|sin)``.

    Every request also gets rows for up to ``ratio - 1`` positions before its chunk: the QSA
    indexer ropes a pooled key at its group's FIRST token, which for a chunked text prompt sharing
    the batch may come from the previous chunk (image prompts are never chunked). Requests without
    ``mrope_positions`` use their token index on all three axes, so their rows equal the rope
    cache rows exactly.
    """
    segments, rows, base = [], [], 0
    for r in reqs:
        pre = min(r.cached_len, ratio - 1)
        start = r.cached_len - pre
        if getattr(r, "mrope_positions", None) is not None:
            seg = r.mrope_positions[:, start : r.device_len]
        else:
            seg = torch.arange(start, r.device_len).expand(3, -1)
        segments.append(seg)
        rows.append(torch.arange(base + pre, base + seg.shape[1], dtype=torch.int32))
        base += seg.shape[1]
    table = mrope_cos_sin(torch.cat(segments, dim=1).to(device), inv_freq, section)
    return torch.cat(rows).to(device), table


__all__ = ["mrope_cos_sin", "mrope_table", "rope_index"]
