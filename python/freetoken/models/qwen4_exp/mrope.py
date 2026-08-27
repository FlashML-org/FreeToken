"""Qwen VL multimodal rotary-position helpers."""

from __future__ import annotations

import itertools

import torch


def build_mrope_positions(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Build exact Qwen3-VL image/text MRoPE positions for one request.

    ``mm_token_type_ids`` uses 0 for text and 1 for image tokens. Video is
    rejected until the server carries its timestamps and grid metadata.
    """
    tokens = input_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    types = mm_token_type_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    grids = image_grid_thw.detach().to(device="cpu", dtype=torch.int64).reshape(-1, 3)
    if types.numel() != tokens.numel():
        raise ValueError(
            "mm_token_type_ids length must match input_ids: "
            f"{types.numel()} != {tokens.numel()}"
        )
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")

    grid_index = 0
    current_position = 0
    pieces: list[torch.Tensor] = []
    for modality, group in itertools.groupby(enumerate(types.tolist()), lambda item: item[1]):
        members = list(group)
        group_len = len(members)
        if modality == 0:
            positions = torch.arange(current_position, current_position + group_len)
            pieces.append(positions.view(1, -1).expand(3, -1))
            current_position += group_len
            continue
        if modality != 1:
            raise NotImplementedError(
                "Qwen4-Exp video MRoPE is not enabled; image input is supported"
            )
        if grid_index >= grids.shape[0]:
            raise ValueError("mm_token_type_ids contains more image groups than image_grid_thw")
        grid_t, grid_h, grid_w = (int(value) for value in grids[grid_index].tolist())
        grid_index += 1
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError("image grid height and width must divide by spatial_merge_size")
        llm_t = grid_t
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        expected = llm_t * llm_h * llm_w
        if group_len != expected:
            raise ValueError(
                "image-token group length does not match image_grid_thw: "
                f"{group_len} != {expected}"
            )
        temporal = torch.arange(llm_t)
        height = torch.arange(llm_h) + current_position
        width = torch.arange(llm_w) + current_position
        t_grid, h_grid, w_grid = torch.meshgrid(
            temporal, height, width, indexing="ij"
        )
        vision = torch.stack((t_grid, h_grid, w_grid), dim=0).reshape(3, -1)
        vision[0].add_(current_position)
        pieces.append(vision)
        current_position += max(llm_h, llm_w)

    if grid_index != grids.shape[0]:
        raise ValueError("image_grid_thw contains more images than mm_token_type_ids")
    positions = torch.cat(pieces, dim=1) if pieces else torch.empty((3, 0), dtype=torch.int64)
    delta = int(positions.max().item() + 1 - tokens.numel()) if tokens.numel() else 0
    return positions.contiguous(), delta


__all__ = ["build_mrope_positions"]
