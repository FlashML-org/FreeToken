"""GGML NVFP4 packing helpers shared by native GGUF model loaders."""

from __future__ import annotations

import torch


def nvfp4_parts(t, *, dense: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Split GGML NVFP4 blocks into FreeToken packed values and E4M3 scales.

    GGML stores each 64-value block as 4 FP8-e4m3 scales followed by four 16-value
    groups. Each group is laid out as eight low-nibble codes followed by eight
    high-nibble codes. FreeToken expects adjacent codes in each packed byte, so the
    group halves must be interleaved before returning the packed stream.
    """
    raw = t.packed()
    blocks = raw.reshape(t.rows, -1, 36)
    scales = blocks[:, :, :4].reshape(t.rows, -1).contiguous()

    grouped = blocks[:, :, 4:].reshape(t.rows, -1, 4, 8)
    # GGML stores each 16-value group as [low0..low7, high0..high7]. Restore
    # [low0, high0, low1, high1, ...] for the FreeToken byte layout.
    codes = torch.cat((grouped & 0x0F, grouped >> 4), dim=-1).reshape(t.rows, -1, 16)
    if dense:
        # The dense int32 kernel consumes four pairs as [0,4], [1,5], [2,6],
        # [3,7] within each eight-code half of a 16-code scale group.
        halves = codes.reshape(t.rows, -1, 2, 8)
        codes = torch.stack((halves[..., :4], halves[..., 4:]), dim=-1).reshape(t.rows, -1, 16)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(t.rows, -1).contiguous()
    return packed, scales.view(torch.float8_e4m3fn)


__all__ = ["nvfp4_parts"]