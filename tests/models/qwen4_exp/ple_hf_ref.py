"""HF ground truth for the PLE parity tests, run as a script under the transformers-main venv.

``transformers`` in the serving venv predates ``models/qwen4_exp``, so the reference classes cannot
be imported next to FreeToken. ``test_ple.py`` spawns this file with the reference interpreter:
``python test_ple_hf_ref.py spec.json inputs.npz out.npz``. It holds no tests; every import is
inside ``main`` so pytest can still collect the module.
"""

from __future__ import annotations


def main() -> None:
    import json
    import sys

    import numpy as np
    import torch
    from torch import nn
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding,
        Qwen4ExpTextPLELayer,
    )

    class CaptureEmbedding(nn.Module):
        """Stands in for the table so the hashed ids can be read out of the HF module."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
            self.ids = None

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            self.ids = ids.clone()
            return torch.zeros(*ids.shape, self.weight.shape[1])

    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    data = np.load(sys.argv[2])
    config = Qwen4ExpTextConfig(**spec["config"])
    layer_idx, ple_index = spec["layer_idx"], spec["ple_layer_index"]
    out = {}

    embed = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx, ple_index)
    out["layer_multipliers"] = embed.layer_multipliers.numpy()
    out["ngram_heads_vocab_sizes"] = embed.ngram_heads_vocab_sizes.numpy()
    out["ngram_heads_offsets"] = embed.ngram_heads_offsets.numpy()
    out["padded_vocab_size"] = np.array(embed.ngram_embedding.weight.shape[0])

    capture = CaptureEmbedding(embed.ngram_embedding.embedding_dim)
    embed.ngram_embedding = capture
    embed(torch.as_tensor(data["hash_tokens"]).long(), None)
    out["hash_ids"] = capture.ids.numpy()

    layer = Qwen4ExpTextPLELayer(config, layer_idx, ple_index)
    with torch.no_grad():
        for name in ("key_proj", "value_proj", "norm_key", "norm_query", "norm_conv"):
            getattr(layer, name).weight.copy_(torch.as_tensor(data[name]))
        layer.conv1d.weight.copy_(torch.as_tensor(data["conv1d"]))
        layer.ple_embedding.ngram_embedding.weight.copy_(torch.as_tensor(data["table"]))
        out["layer_out"] = layer(
            torch.as_tensor(data["hidden"]).float(),
            torch.as_tensor(data["layer_tokens"]).long(),
            None,
        ).numpy()
    np.savez(sys.argv[3], **out)


if __name__ == "__main__":
    main()
