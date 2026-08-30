from __future__ import annotations

from types import SimpleNamespace

import torch


def test_glm5_next_weight_rename_covers_checkpoint_layout():
    from freetoken.models.glm5_next.weight import _rename

    prefix = "model.language_model.layers.3"
    assert _rename(f"{prefix}.hc_attn_fn") == "model.layers.3.attn_hc.fn"
    assert _rename(f"{prefix}.hc_ffn_base") == "model.layers.3.ffn_hc.base"
    assert _rename(f"{prefix}.mlp.gate.e_score_correction_bias") == (
        "model.layers.3.mlp.e_score_correction_bias"
    )
    assert _rename(f"{prefix}.self_attn.q_a_proj.weight") == (
        "model.layers.3.self_attn.q_a_proj.weight"
    )
    assert _rename("model.visual.patch_embed.proj.weight") is None
    assert _rename(f"{prefix}.mlp.experts.7.gate_proj.weight") is None
    assert _rename(f"{prefix}.mlp.shared_experts.gate_proj.weight_scale") is None
    assert _rename("model.language_model.layers.45.shared_head.norm.weight") is None


def test_glm5_next_fuses_kda_convs_per_layer_in_qkv_order():
    from freetoken.models.glm5_next.weight import _try_fuse_kda_conv

    buf: dict[str, dict[int, torch.Tensor]] = {}
    base0 = "model.layers.0.self_attn"
    base1 = "model.layers.1.self_attn"
    q = torch.full((2, 1, 4), 1.0)
    k = torch.full((2, 1, 4), 2.0)
    v = torch.full((2, 1, 4), 3.0)

    assert _try_fuse_kda_conv(f"{base0}.q_conv1d.weight", q, buf) == ()
    assert _try_fuse_kda_conv(f"{base1}.v_conv1d.weight", v, buf) == ()
    assert _try_fuse_kda_conv(f"{base0}.v_conv1d.weight", v, buf) == ()
    fused = _try_fuse_kda_conv(f"{base0}.k_conv1d.weight", k, buf)

    assert fused is not None and fused != ()
    name, tensor = fused
    assert name == f"{base0}.conv1d.weight"
    assert tensor.shape == (6, 1, 4)
    assert tensor[:, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert f"{base1}.conv1d.weight" in buf
    assert _try_fuse_kda_conv(f"{base0}.q_proj.weight", q, buf) is None


def test_glm5_next_expert_bank_mapping_excludes_appended_mtp_layer():
    from freetoken.models.glm5_next.weight import _SOURCE_SPEC

    config = SimpleNamespace(first_k_dense_replace=3, num_layers=45)
    assert _SOURCE_SPEC.layer_to_bank(2, config) is None
    assert _SOURCE_SPEC.layer_to_bank(3, config) == 0
    assert _SOURCE_SPEC.layer_to_bank(44, config) == 41
    assert _SOURCE_SPEC.layer_to_bank(45, config) is None


def test_glm5_next_weight_loader_uses_safe_open_keys(monkeypatch):
    from freetoken.models.glm5_next import weight

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["model.language_model.norm.weight"]

        def get_tensor(self, name):
            assert name == "model.language_model.norm.weight"
            return torch.ones(2)

    monkeypatch.setattr(weight, "get_tp_info", lambda: SimpleNamespace(size=1))
    monkeypatch.setattr(
        weight, "iter_weight_files", lambda path: ["weights.safetensors"]
    )
    monkeypatch.setattr(
        weight.safetensors, "safe_open", lambda *args, **kwargs: FakeSafeOpen()
    )
    monkeypatch.setattr(weight, "drop_page_cache", lambda path: None)

    loaded = list(
        weight.iter_weights(
            "/model",
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    assert len(loaded) == 1
    assert loaded[0][0] == "model.norm.weight"
    assert torch.equal(loaded[0][1], torch.ones(2))
