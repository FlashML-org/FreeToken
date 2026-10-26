from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, steps: int = 1, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs * steps,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            # constant under a graph: every request has exactly `steps` query rows
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device) * steps,
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
